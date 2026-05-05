from .gpt import GPTWrapper
from .qwen import OpenAICompatibleWrapper

LLM_WRAPPER = dict(gpt=GPTWrapper, openai_compatible=OpenAICompatibleWrapper)