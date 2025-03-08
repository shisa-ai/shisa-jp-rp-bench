import choix
import numpy as np
from typing import List, Tuple
import pandas as pd
import click
import glob
import json
import re
import os
import shutil

class LLMRanker:
    def __init__(self):
        self.llm_to_idx = {}  # Maps LLM names to consecutive integers
        self.idx_to_llm = {}  # Reverse mapping
        self.n_items = 0
        self.params = None
        self.wins_count = {}  # Track raw number of wins for each LLM
        self.total_matches = {}  # Track total number of matches for each LLM

    def process_comparisons(self, comparisons: List[Tuple[str, str, str]]):
        """
        Process raw comparison data into format needed for choix.
        
        Args:
            comparisons: List of tuples (llm1, llm2, winner)
                        where winner is either "llm1" or "llm2"
        """
        # First, build the mapping of LLM names to integers
        unique_llms = set()
        for llm1, llm2, _ in comparisons:
            unique_llms.add(llm1)
            unique_llms.add(llm2)
        
        self.llm_to_idx = {llm: idx for idx, llm in enumerate(unique_llms)}
        self.idx_to_llm = {idx: llm for llm, idx in self.llm_to_idx.items()}
        self.n_items = len(self.llm_to_idx)
        
        # Initialize wins count for each LLM
        self.wins_count = {llm: 0 for llm in unique_llms}
        self.total_matches = {llm: 0 for llm in unique_llms}
        
        # Convert comparisons to format needed by choix
        processed_comparisons = []
        for llm1, llm2, winner in comparisons:
            idx1 = self.llm_to_idx[llm1]
            idx2 = self.llm_to_idx[llm2]
            # Track wins and matches
            self.wins_count[winner] += 1
            self.total_matches[llm1] += 1
            self.total_matches[llm2] += 1
            if winner == llm1:
                processed_comparisons.append((idx1, idx2))
            else:
                processed_comparisons.append((idx2, idx1))
                
        return processed_comparisons

    def fit(self, comparisons: List[Tuple[str, str, str]]):
        """
        Fit the Bradley-Terry model to the comparison data.
        """
        processed_data = self.process_comparisons(comparisons)
        self.params = choix.opt_pairwise(self.n_items, processed_data)
        
    def get_rankings(self) -> pd.DataFrame:
        """
        Returns a DataFrame with LLMs ranked by their scores and win counts.
        """
        if self.params is None:
            raise ValueError("Must fit model before getting rankings")
        
        '''
        See more about EN & LT: https://chatgpt.com/share/67b34c25-61c8-8012-8667-17077284d92a
        '''

        # 1) Exponential & Normalize (EN)
        exp_params = np.exp(self.params)
        sum_exp = np.sum(exp_params)
        en_scores = exp_params / sum_exp  # 0-1 scale
        en_scores_0_10 = en_scores * 10   # 0-10 scale if desired

        # 2) Logistic Transform (LT)
        # Shift so the average model has param=0 => logistic transform is 0.5 on average
        mean_param = np.mean(self.params)
        shifted_params = self.params - mean_param
        lt_scores = 1.0 / (1.0 + np.exp(-shifted_params))  # 0-1 scale
        lt_scores_0_10 = lt_scores * 10                    # 0-10 scale if desired
            
        rankings = pd.DataFrame({
            'llm': [self.idx_to_llm[i] for i in range(self.n_items)],
            'score': self.params,
            'wins': [self.wins_count[self.idx_to_llm[i]] for i in range(self.n_items)],
            'total_matches': [self.total_matches[self.idx_to_llm[i]] for i in range(self.n_items)],
            'EN': en_scores_0_10,
            'LT': lt_scores_0_10,
        })
        return rankings.sort_values('score', ascending=False).reset_index(drop=True)
    
    def predict_winner_probability(self, llm1: str, llm2: str) -> float:
        """
        Predict probability that llm1 will win against llm2.
        """
        if self.params is None:
            raise ValueError("Must fit model before making predictions")
            
        idx1 = self.llm_to_idx[llm1]
        idx2 = self.llm_to_idx[llm2]
        prob, _ = choix.probabilities((idx1, idx2), self.params)
        return prob

@click.command()
@click.option('--target-model', '-m', required=False, help='Name of the model being evaluated')
@click.option('--judge-model', '-j', required=False, help='Name of the model did the judging')
@click.option('--temp-dir', help='Temporary directory for job-specific files')
def main(target_model, judge_model, temp_dir):
    # Set up job-specific directories
    analysis_dir = os.path.join(temp_dir, "analysis") if temp_dir else "analysis"
    temp_scores_dir = os.path.join(temp_dir, "scores") if temp_dir else "scores"

    # Always create the global scores directory as well for final output
    global_scores_dir = "scores"  # This is the directory your caller expects
    os.makedirs(global_scores_dir, exist_ok=True)
    # Create temp scores directory if needed
    if temp_dir:
        os.makedirs(temp_scores_dir, exist_ok=True)
    
    # Always load base set comparisons first
    comparisons = []
    # Update file paths to use job-specific directories
    safe_judge_name = judge_model.replace("/", "__")
    base_files = [f'analysis/base_set.{safe_judge_name}.jsonl']
    if base_files:
        print("\nProcessing base set comparisons...")
        for base_file in base_files:
            comparisons.extend(load_comparisons_from_file(base_file))
    
    # If target model and judge model are specified, load those comparisons too
    if target_model and judge_model:
        safe_model_name = target_model.replace("/", "__")
        safe_judge_name = judge_model.replace("/", "__")
        file_path = os.path.join(analysis_dir, f'{safe_model_name}.{safe_judge_name}.jsonl')
        
        if os.path.exists(file_path):
            print(f"\nProcessing {file_path}...")
            comparisons.extend(load_comparisons_from_file(file_path))
        else:
            print(f"Analysis file not found: {file_path}")
            if not comparisons:  # If we don't even have base comparisons
                exit(1)
    
    if not comparisons:
        print("No valid comparisons found in any files")
        exit(1)
            
    # Initialize and fit the model
    ranker = LLMRanker()
    ranker.fit(comparisons)
    
    # Get rankings
    rankings = ranker.get_rankings()
    print("\nRankings:")
    print(rankings)
    
    # Print win counts
    print("\nRaw win counts:")
    for llm, wins in sorted(ranker.wins_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{llm}: {wins} wins")
    
    # Only save files if both model names are provided
    if target_model and judge_model:
        # Save rankings with safe model names
        safe_model_name = target_model.replace("/", "__")

        # Save score to temp and global locations
        temp_scores_file = os.path.join(temp_scores_dir, f'{safe_model_name}_rp_bench_scores.jsonl')
        rankings.to_json(temp_scores_file, orient='records', lines=True)
        
        global_scores_file = os.path.join(global_scores_dir, f'{safe_model_name}_rp_bench_scores.jsonl')
        rankings.to_json(global_scores_file, orient='records', lines=True)
        print(f"\nScores saved to: {global_scores_file}")
        
        # Copy answers to global locations
        global_answers_file = os.path.join(global_scores_dir, f'{safe_model_name}_rp_bench_answers.jsonl')
        shutil.copy(file_path, global_answers_file)
        print(f"Results saved to: {global_answers_file}")

def load_comparisons_from_file(file_path):
    """Helper function to load comparisons from a file."""
    comparisons = []
    # Check if this is a base set file
    is_base_file = 'base_set.' in file_path
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' not in data or 'llm_b' not in data or 'analysis' not in data:
                    continue
                
                # Extract the winner from analysis field
                match = re.search(r'<answer>(.*?)</answer>', data['analysis'])
                if not match:
                    continue
                        
                answer_content = match.group(1)
                # Remove anything that isn't an ASCII letter
                cleaned_answer = ''.join(c for c in answer_content if c.isalpha())
                
                if not cleaned_answer:
                    continue
                        
                cleaned_answer = cleaned_answer.lower()
                
                '''
                In order to have proper handling of models that appear in the base set, we prefix base set models with a 'base__'

                The base model is always llm2 for our target model comparison file so we have to prefix 'base__' as well there
                '''
                # Add prefix to model names if from base set
                if is_base_file:
                    llm1 = f"base__{data['llm_a']}"
                    llm2 = f"base__{data['llm_b']}"
                else:
                    llm1 = data['llm_a']
                    llm2 = f"base__{data['llm_b']}"

                
                if cleaned_answer == 'a':
                    winner = llm1
                elif cleaned_answer == 'b':
                    winner = llm2
                else:
                    print(f"Error: Invalid answer content in <answer> tag: {answer_content}")
                    continue
                        
                comparisons.append((llm1, llm2, winner))
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON line encountered")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    return comparisons

if __name__ == "__main__":
    main()
