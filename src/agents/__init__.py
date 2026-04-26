from .alfworld import ReflectAgent, CDMemAgent, AutoguideAgent, ExpelAgent, ReactAgent
from .scienceworld import CDMemAgentSC, ReflectAgentSC, ReactAgentSC

# 双层嵌套dict
AGENT = dict(alfworld=dict(reflect=ReflectAgent, cdmem=CDMemAgent, autoguide=AutoguideAgent, expel=ExpelAgent, react=ReactAgent),
             scienceworld=dict(reflect=ReflectAgentSC, react=ReactAgentSC, cdmem=CDMemAgentSC))