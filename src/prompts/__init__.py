from .alfworld import CDMemPromptBuilder, ReflectPromptBuilder, AutoguidePromptBuilder, ExpelPromptBuilder
from .scienceworld import CDMemPromptBuilderSC, ReflectPromptBuilderSC

PROMPT_BUILDER = dict(alfworld=dict(reflect=ReflectPromptBuilder, cdmem=CDMemPromptBuilder, cdmem_autogen=CDMemPromptBuilder, autoguide=AutoguidePromptBuilder, expel=ExpelPromptBuilder, react=ReflectPromptBuilder),
                      scienceworld=dict(reflect=ReflectPromptBuilderSC, cdmem=CDMemPromptBuilderSC))
