import os
import sys
from openai import OpenAI
import json
from regex import F
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

Model = Literal["qwen/qwen3-32b", "qwen/qwen-2.5-72b-instruct"]
ChatModel = Literal["qwen/qwen3-32b", "qwen/qwen-2.5-72b-instruct"]


class QwenWrapper:
    def __init__(self, model: Model):
        self.client = OpenAI(
            base_url=os.getenv(
                'OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None,
            api_key=os.getenv('OPENAI_API_KEY'),
        )
        self.model = model

    def __call__(self, prompt: str, stop: List[str] = None, max_tokens: int = 256, mode: str = 'chat', model=None, sys_msg=None, use_json=False) -> str:
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
                # 添加对None值的兼容
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

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_chat(self, prompt: str, model: ChatModel, max_tokens: int, temperature: float = 0.0, sys_msg=None,
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
        if use_json:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={'type': 'json_object'},
                extra_body={"reasoning": {"enabled": False}}
                # 不开启思考
            )
            content = response.choices[0].message.content
            #
            return content if content is not None else ""
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop_strs,
                temperature=temperature,
                extra_body={"reasoning": {"enabled": False}}

            )
            content = response.choices[0].message.content
            print(f'<🏃ModelOutput>: {content}\n<🏃End>')
            return content if content is not None else ""

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(self, prompt: str, model: ChatModel, max_tokens: int, temperature: float = 0.0,
                       stop_strs: Optional[List[str]] = None) -> str:
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
            extra_body={"reasoning": {"enabled": False}}
            # 不开启思考
        )
        content = response.choices[0].message.content
        # 兼容None值
        return content if content is not None else ""
