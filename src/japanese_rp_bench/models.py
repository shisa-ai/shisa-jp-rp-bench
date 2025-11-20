# models.py

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import google.genai as genai
from google.genai import types as genai_types


# 各種モデルをロードするための抽象化された関数
def load_model(
    model_name: str,
    inference_method: str,
    tensor_parallel_size: int = 1,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    指定されたモデルをロードする関数

    Args:
        model_name (str): モデル名
        inference_method (str): 推論方法
        tensor_parallel_size (int, optional): テンソル並列数. Defaults to 1.
        cache_dir (Optional[str], optional): キャッシュディレクトリ. Defaults to None.

    Returns:
        Tuple[Any, Any]: モデルとトークナイザーのタプル
    """
    # OpenAI APIを使用する場合
    if inference_method == "openai_api":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or None
        if not api_key:
            raise ValueError(
                "openai api key is not set, please set OPENAI_API_KEY in environment variable."
            )
        client = OpenAI(api_key=api_key)
        return client, None

    # OpenAI互換のAPIの場合
    elif inference_method == "openai_compatible_api":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or None
        if not api_key:
            raise ValueError(
                "openai compatible api key is not set, please set OPENAI_COMPATIBLE_API_KEY in environment variable."
            )
        api_url = os.getenv("OPENAI_COMPATIBLE_API_URL") or None
        if not api_url:
            raise ValueError(
                "openai compatible api url is not set, please set OPENAI_COMPATIBLE_API_URL in environment variable."
            )
        client = OpenAI(api_key=api_key, base_url=api_url)
        return client, None

    # AnthropicのAPIの場合
    elif inference_method == "anthropic_api":
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or None
        if not api_key:
            raise ValueError(
                "anthropic api key is not set, please set ANTHROPIC_API_KEY in environment variable."
            )
        client = Anthropic(api_key=api_key)
        return client, None

    # Amazon BedrockのAnthropic APIの場合
    elif inference_method == "aws_anthropic_api":
        from anthropic import AnthropicBedrock
        aws_access_key = os.getenv("AWS_ACCESS_KEY") or None
        aws_secret_key = os.getenv("AWS_SECRET_KEY") or None
        if not aws_access_key:
            raise ValueError(
                "AWS access key is not set, please set AWS_ACCESS_KEY in environment variable."
            )
        if not aws_secret_key:
            raise ValueError(
                "AWS secret key is not set, please set AWS_SECRET_KEY in environment variable."
            )
        client = AnthropicBedrock(
            aws_access_key=aws_access_key, aws_secret_key=aws_secret_key
        )
        return client, None

    # CohereのAPIの場合
    elif inference_method == "cohere_api":
        from cohere import Client
        api_key = os.getenv("COHERE_API_KEY") or None
        if not api_key:
            raise ValueError(
                "cohere api key is not set, please set COHERE_API_KEY in environment variable."
            )
        client = Client(api_key=api_key)
        return client, None

    # Google AI APIの場合 (Gemini via google-genai)
    elif inference_method == "google_api":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
        if not api_key:
            raise ValueError(
                "google api key is not set, please set GOOGLE_API_KEY or GEMINI_API_KEY in environment variable."
            )
        client = genai.Client(api_key=api_key)
        return client, None

    # MistralAI APIの場合
    elif inference_method == "mistralai_api":
        from mistralai import Mistral
        api_key = os.getenv("MISTRAL_API_KEY") or None
        if not api_key:
            raise ValueError(
                "mistral api key is not set, please set MISTRAL_API_KEY in environment variable."
            )
        client = Mistral(api_key=api_key)
        return client, None

    # LiteLLMを使用する場合
    elif inference_method == "litellm":
        from .helpers.llmcaller.litellm_caller import LiteLLMCaller

        client = LiteLLMCaller(
            model=model_name,
            api_base="http://localhost:8000/v1",
        )
        litellm.set_verbose= True
        os.environ['LITELLM_LOG'] = 'DEBUG'
        return client, None
        
    # Hugging Faceのモデルを使用する場合
    elif inference_method == "huggingface":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading Hugging Face model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer

    # vLLMを使ってローカルで推論する場合
    elif inference_method == "vllm":
        from vllm import LLM, SamplingParams
        print("Loading vLLM model...")
        model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=cache_dir,
            enable_prefix_caching=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        return model, tokenizer

    # transformersを使ってローカルで推論する場合
    elif inference_method == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        return model, tokenizer
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")

# 各種モデルから応答を生成する関数
def generate_response(
    model: Any,
    tokenizer: Optional[Any],
    model_name: str,
    inference_method: str,
    system_prompt: str,
    conversations: List[Dict[str, str]],
    low_context: bool = False,
    ultra_low_context: bool = False,
) -> str:
    # OpenAIやOpenAI互換のAPIの場合
    if inference_method in ["openai_api", "openai_compatible_api"]:
        if "o1" in model_name:
            messages = []
            # o1はシステムプロンプトをサポートしていないのでシステムプロンプトと最初の会話を結合
            first_conversation = conversations[0]
            messages.append(
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{first_conversation['content']}",
                }
            )
            # 残りの会話を追加
            messages.extend(conversations[1:])
            result = model.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=128 if ultra_low_context else (256 if low_context else 1024),
                temperature=0.7,
            )
        else:
            # o1以外のモデルの場合
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversations)
            result = model.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=128 if ultra_low_context else (256 if low_context else 1024),
                temperature=0.7,
            )
        response = result.choices[0].message.content.strip()

    # AnthropicのAPIの場合
    elif inference_method == "anthropic_api":
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = []
        for i, conversation in enumerate(conversations):
            # Anthropicのprompt cachingは最大4か所まで
            if i < 3:
                messages.append(
                    {
                        "role": conversation["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": conversation["content"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": conversation["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": conversation["content"],
                            }
                        ],
                    }
                )
        result = model.messages.create(
            model=model_name,
            system=system,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        response = result.content[0].text.strip()

    # Amazon BedrockのAnthropic APIの場合
    elif inference_method == "aws_anthropic_api":
        messages = []
        for conversation in conversations:
            messages.append(
                {
                    "role": conversation["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": conversation["content"],
                        }
                    ],
                }
            )
        result = model.messages.create(
            model=model_name,
            system=system_prompt,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response = result.content[0].text.strip()

    # CohereのAPIの場合
    elif inference_method == "cohere_api":
        preamble = system_prompt
        chat_history = []
        for i, conversation in enumerate(conversations):
            if i == len(conversations) - 1:
                message = conversation["content"]
            else:
                if i % 2 == 0:
                    chat_history.append(
                        {"role": "User", "message": conversation["content"]}
                    )
                else:
                    chat_history.append(
                        {"role": "Chatbot", "message": conversation["content"]}
                    )
        result = model.chat(
            model=model_name,
            message=message,
            chat_history=chat_history,
            preamble=preamble,
            temperature=0.7,
            max_tokens=1024,
        )
        response = result.text.strip()

    # Google AI APIの場合 (Gemini via google-genai)
    elif inference_method == "google_api":
        # `model` is a google.genai.Client instance from load_model
        client = model
        max_tokens = 128 if ultra_low_context else (256 if low_context else 1024)

        generation_config = genai_types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )

        # Flatten system prompt + conversation turns into a single text prompt
        message_lines: List[str] = [system_prompt, ""]
        for conv in conversations:
            role = conv.get("role", "user")
            content = conv.get("content", "")
            message_lines.append(f"{role}: {content}")
        message = "\n".join(message_lines)

        result = client.models.generate_content(
            model=model_name,
            contents=message,
            config=generation_config,
        )
        response = result.text.strip()

    # MistralAI APIの場合
    elif inference_method == "mistralai_api":
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations)
        result = model.chat.complete(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response = result.choices[0].message.content.strip()

    # vLLMを使ってローカルで推論する場合
    elif inference_method == "vllm":
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations)
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
        )
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        result = model.generate(input_text, sampling_params)
        response = result[0].outputs[0].text.strip()

    # transformersを使ってローカルで推論する場合
    elif inference_method == "transformers":
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversations)
        return_output =  tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True 
        ).to(model.device)
        input_ids = return_output["input_ids"]
        attention_mask = return_output["attention_mask"]
        result = model.generate(input_ids, attention_mask=attention_mask, temperature=0.7, max_new_tokens=1024)
        response = tokenizer.decode(
            result.tolist()[0][input_ids.size(1) :], skip_special_tokens=True
        ).strip()

    else:
        raise ValueError(f"Unknown inference method: {inference_method}")

    return response
