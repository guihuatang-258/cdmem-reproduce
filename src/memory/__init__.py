from .alfworld import ReflectShortMemory, ReflectLocalMemory, CDMemShortMemory, CDMemLocalMemory, CDMemGlobalMemory
from .alfworld import AutoguideShortMemory, AutoguideLocalMemory, AutoguideGlobalMemory
from .alfworld import ExpelShortMemory, ExpelLocalMemory, ExpelGlobalMemory
from .scienceworld import ReflectShortMemorySC, ReflectLocalMemorySC, CDMemShortMemorySC, CDMemLocalMemorySC, CDMemGlobalMemorySC

SHORT_MEMORY = dict(alfworld=dict(reflect=ReflectShortMemory, cdmem=CDMemShortMemory, cdmem_autogen=CDMemShortMemory, autoguide=AutoguideShortMemory, expel=ExpelShortMemory, react=ReflectShortMemory),
                    scienceworld=dict(reflect=ReflectShortMemorySC, cdmem=CDMemShortMemorySC))
LOCAL_MEMORY = dict(alfworld=dict(reflect=ReflectLocalMemory, cdmem=CDMemLocalMemory, cdmem_autogen=CDMemLocalMemory, autoguide=AutoguideLocalMemory, expel=ExpelLocalMemory, react=ReflectLocalMemory),
                    scienceworld=dict(reflect=ReflectLocalMemorySC, cdmem=CDMemLocalMemorySC))
GLOBAL_MEMORY = dict(alfworld=dict(cdmem=CDMemGlobalMemory, cdmem_autogen=CDMemGlobalMemory, reflect=CDMemGlobalMemory, autoguide=AutoguideGlobalMemory, expel=ExpelGlobalMemory, react=CDMemGlobalMemory),
                    scienceworld=dict(reflect=CDMemGlobalMemorySC, cdmem=CDMemGlobalMemorySC))
