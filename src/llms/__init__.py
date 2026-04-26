from .gpt import GPTWrapper
from .qwen import QwenWrapper

LLM_WRAPPER = dict(gpt=GPTWrapper, qwen=QwenWrapper)