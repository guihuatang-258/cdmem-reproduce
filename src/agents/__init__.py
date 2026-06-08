from .alfworld import ReflectAgent, CDMemAgent, CDMemAutoGenAgent, AutoguideAgent, ExpelAgent, ReactAgent
from .scienceworld import CDMemAgentSC, ReflectAgentSC, ReactAgentSC

# 双层嵌套dict
AGENT = dict(alfworld=dict(reflect=ReflectAgent, cdmem=CDMemAgent, cdmem_autogen=CDMemAutoGenAgent, autoguide=AutoguideAgent, expel=ExpelAgent, react=ReactAgent),
             scienceworld=dict(reflect=ReflectAgentSC, react=ReactAgentSC, cdmem=CDMemAgentSC))
