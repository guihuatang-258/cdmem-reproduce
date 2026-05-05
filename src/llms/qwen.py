import os
import sys
from openai import OpenAI
import json
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from typing import Optional, List


class OpenAICompatibleWrapper:
    """
    通用的 OpenAI 兼容模型 wrapper
    支持所有符合 OpenAI API 格式的模型服务
    模型名称格式: <provider>/<model-name>
    例如: "qwen/qwen3.5-flash-02-23", "openai/gpt-4", "anthropic/claude-3"
    """
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None, 
                 disable_reasoning: bool = True):
        self.client = OpenAI(
            base_url=base_url or (os.getenv('OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None),
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
        )
        self.model = model
        self.disable_reasoning = disable_reasoning

    def __call__(self, prompt: str, stop: List[str] = None, max_tokens: int = 256, mode: str = 'chat', 
                 model=None, sys_msg=None, use_json=False) -> str:
        if not model:
            model = self.model
        try:
            cur_try = 0
            while cur_try < 6:
                if mode == "chat":
                    text = self.get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2,
                                         stop_strs=stop, max_tokens=max_tokens, sys_msg=sys_msg, use_json=use_json)
                elif mode == "complete":
                    text = self.get_completion(
                        prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop, max_tokens=max_tokens)
                else:
                    raise ValueError(
                        f"Invalid mode: {mode}, mode must be 'chat' or 'complete'.")
                if text is None:
                    print("❗️None Text")
                    cur_try += 1
                    continue
                if len(text.strip()) >= 5:
                    if use_json:
                        return json.loads(text)
                    else:
                        return text
                cur_try += 1
            print("⚠️Return Empty")
            return ""
        except Exception as e:
            print(prompt)
            print(e)
            import sys
            sys.exit(1)

    def _get_extra_body(self) -> Optional[dict]:
        """获取额外的请求参数，不同模型可能需要不同的配置"""
        if self.disable_reasoning:
            return {"reasoning": {"enabled": False}}
        return None

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_chat(self, prompt: str, model: str, max_tokens: int, temperature: float = 0.0, sys_msg=None,
                 use_json=False, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
        if sys_msg:
            messages = [
                {
                    "role": "system",
                    "content": sys_msg
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        
        extra_body = self._get_extra_body()
        
        if use_json:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={'type': 'json_object'},
                extra_body=extra_body
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        else:
            # print("❗️执行了get_chat")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop_strs,
                temperature=temperature,
                extra_body=extra_body
            )
            content = response.choices[0].message.content
            return content if content is not None else ""

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(self, prompt: str, model: str, max_tokens: int, temperature: float = 0.0,
                       stop_strs: Optional[List[str]] = None) -> str:
        extra_body = self._get_extra_body()
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strs,
            extra_body=extra_body
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
