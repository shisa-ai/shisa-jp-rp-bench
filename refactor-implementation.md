# Japanese RP Bench Refactor Implementation Notes

## Current Architecture

The system consists of several components that work together to evaluate Japanese roleplay conversations:

1. `run_japanese_rp_bench.sh`: Main orchestration script
2. `generate_shootout_data.py`: Generates conversation pairs for comparison
3. `conversation_comparer_any_model.py`: Compares conversations using a judge model
4. `choix_analyzer.py`: Analyzes results using Bradley-Terry model
5. Third-party `japanese-rp-bench` library (out of scope for refactor)

## Issues Identified

### 1. Configuration Management

#### Current Issues:
- Hardcoded paths across multiple files
- Hardcoded configuration values
- Inconsistent configuration between files
- No central configuration management

#### Suggested Changes:
```python
# Create a central config.py
class Config:
    # Directories
    BASE_CONVERSATIONS_DIR = "base_conversations"
    CONVERSATIONS_DIR = "conversations"
    ANALYSIS_DIR = "analysis"
    SCORES_DIR = "scores"
    
    # Dataset
    DATASET_NAME = "Aratako/Japanese-RP-Bench-testdata-SFW"
    ROWS_TO_USE = 15
    MAX_ROWS = 30
    
    # Files
    PROMPT_FILE = "prompt.txt"
    BASE_PAIRS_FILE = "base_conversation_pairs.jsonl"
    LATEST_PAIRS_FILE = "latest_conversation_pairs.jsonl"
    
    # Allow overrides via environment variables
    @classmethod
    def load(cls):
        for key in dir(cls):
            if key.isupper():
                env_val = os.getenv(f"BENCH_{key}")
                if env_val:
                    setattr(cls, key, env_val)
```

### 2. Directory Structure

#### Current Issues:
- No validation of required directory structure
- Inconsistent directory references
- No clear documentation of expected structure

#### Suggested Changes:
- Implement directory validation
- Document required structure:
```
/
├── base_conversations/     # Base model conversations
├── conversations/         # New model conversations
├── analysis/             # Comparison results
├── scores/              # Final scores and rankings
├── configs/             # Configuration files
└── prompt.txt           # Prompt template
```

### 3. Error Handling & Validation

#### Current Issues:
- Inconsistent error handling across files
- Limited input validation
- No validation of dataset size vs rows_to_use
- No validation of prompt.txt existence

#### Suggested Changes:
```python
class ValidationError(Exception):
    pass

def validate_environment():
    """Validate all required components exist"""
    validate_directories()
    validate_files()
    validate_dataset()

def validate_directories():
    for dir_path in [Config.BASE_CONVERSATIONS_DIR, ...]:
        if not os.path.exists(dir_path):
            raise ValidationError(f"Required directory not found: {dir_path}")
```

### 4. Logging & Monitoring

#### Current Issues:
- Inconsistent logging approaches
- Limited debugging information
- No structured logging format

#### Suggested Changes:
```python
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bench.log'),
            logging.StreamHandler()
        ]
    )
```

### 5. Code Organization

#### Current Issues:
- Duplicate code for file handling
- No shared utilities
- Vestigial code (commented vLLM settings, unused methods)
- Inconsistent code style

#### Suggested Changes:
- Create shared utilities module:
```python
# utils.py
def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file with error handling"""
    
def save_jsonl(data: List[dict], path: str):
    """Save data to JSONL file"""
    
def format_model_name(name: str) -> str:
    """Consistently format model names"""
```

### 6. Documentation

#### Current Issues:
- Limited inline documentation
- No API documentation
- No clear process documentation
- Evaluation criteria not referenced in code

#### Suggested Changes:
- Add comprehensive docstrings
- Create process documentation
- Document API interfaces
- Include evaluation criteria references

## Suggested Clean Room Implementation

### 1. Project Structure
```
japanese-rp-bench/
├── config/
│   ├── __init__.py
│   ├── default.py        # Default configuration
│   └── custom.py         # Custom overrides
├── core/
│   ├── __init__.py
│   ├── generator.py      # Conversation pair generator
│   ├── comparer.py       # Conversation comparer
│   └── analyzer.py       # Results analyzer
├── utils/
│   ├── __init__.py
│   ├── file_utils.py     # File operations
│   ├── validation.py     # Validation functions
│   └── logging.py        # Logging setup
├── scripts/
│   └── run_bench.sh      # Main script
└── tests/
    └── ...               # Test files
```

### 2. Configuration Management
- Use environment variables for sensitive data
- Support YAML configuration files
- Allow command-line overrides
- Implement configuration validation

### 3. Error Handling
- Implement proper exception hierarchy
- Add comprehensive input validation
- Provide detailed error messages
- Include error recovery mechanisms

### 4. Logging
- Structured logging format
- Support different log levels
- Include performance metrics
- Add debug information

### 5. Testing
- Unit tests for core functionality
- Integration tests for full flow
- Configuration testing
- Error handling testing

## Next Steps

1. Review and prioritize issues
2. Create detailed implementation plan
3. Implement changes incrementally
4. Add comprehensive tests
5. Update documentation

Note: The third-party `japanese-rp-bench` library should be treated as a black box and not modified to avoid maintenance overhead.