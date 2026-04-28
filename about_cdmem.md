根据代码分析，我来详细对比这三种记忆系统的区别和联系：

## 三种记忆系统的区别

### 1. **ShortMemory（短时记忆）**
**位置**: [cdmem_memory.py:L9-L33](file:///home/peter/git-repository/CDMem/src/memory/alfworld/cdmem_memory.py#L9-L33)

**特点**：
- **生命周期最短**：每次 `run_trajectory` 开始时调用 `reset()` 清空
- **存储内容**：当前轨迹的 action-observation 交互对
- **数据结构**：简单的列表 `List[Dict[str, str]]`
- **作用范围**：单次任务执行过程
- **召回方式**：按顺序返回所有历史交互记录

```python
# 存储格式
{'label': 'action', 'value': 'go to countertop 1'}
{'label': 'observation', 'value': 'On the countertop 1...'}
```

### 2. **LocalMemory（本地记忆）**
**位置**: [cdmem_memory.py:L35-L63](file:///home/peter/git-repository/CDMem/src/memory/alfworld/cdmem_memory.py#L35-L63)

**特点**：
- **生命周期**：整个运行过程持久化，按 trial 记录
- **存储内容**：每个环境的专家轨迹和反思记录
- **数据结构**：每个环境独立的字典，包含 `reflection` 列表
- **作用范围**：按环境索引（env_idx）隔离
- **召回方式**：返回该环境的历史反思（最多保留最近3条）

```python
# 存储格式
{
    'name': 'env_0',
    'reflection': ['反思内容1', '反思内容2', ...],
    'is_success': True/False,
    'skip': True/False
}
```

### 3. **GlobalMemory（全局记忆）**
**位置**: [cdmem_memory.py:L65-L231](file:///home/peter/git-repository/CDMem/src/memory/alfworld/cdmem_memory.py#L65-L231)

**特点**：
- **生命周期最长**：跨 trial、跨环境持久化
- **存储内容**：两类记忆
  - **env_memory**：环境知识（容器功能、物品位置等）
  - **task_memory**：任务经验（行动指导、成功/失败经验）
- **数据结构**：字典结构，支持向量检索（ChromaDB）
- **作用范围**：全局共享，所有环境可用
- **召回方式**：基于环境描述和任务类型检索相关知识

```python
# env_memory 结构
{
    '环境描述': {
        'known_obs': '容器功能总结',
        'increment_traj': [...],
        'all_traj': [...]
    }
}

# task_memory 结构
{
    'pick_and_place': {
        'success': {'action_guidance': '经验总结', ...},
        'fail': {'action_guidance': '失败教训', ...}
    }
}
```

---

## 三种记忆系统的联系

### **数据流转关系**

```
ShortMemory (当前交互)
    ↓ 轨迹完成后
LocalMemory (反思更新)
    ↓ 总结提取
GlobalMemory (知识沉淀)
```

### **具体协作流程**

1. **推理阶段** ([cdmem.py:L140-L185](file:///home/peter/git-repository/CDMem/src/agents/alfworld/cdmem.py#L140-L185))
   ```python
   # 同时使用三种记忆构建 prompt
   short_memories = self.short_memory.recall()        # 当前交互历史
   local_memories = self.local_memory.recall(env_idx) # 该环境的反思
   known_obs, action_guidance = self.global_memory.recall(...) # 全局知识
   ```

2. **更新阶段** ([cdmem.py:L119-L136](file:///home/peter/git-repository/CDMem/src/agents/alfworld/cdmem.py#L119-L136))
   ```python
   # 轨迹完成后
   expert_trajectory = self.update_local_memory(...)  # LLM 生成专家轨迹
   self.local_memory.add(env_idx, expert_trajectory)  # 存入本地记忆
   self.update_global_memory(...)                      # 提取总结存入全局记忆
   ```

3. **GlobalMemory 的增量更新机制** ([cdmem_memory.py:L80-L152](file:///home/peter/git-repository/CDMem/src/memory/alfworld/cdmem_memory.py#L80-L152))
   - 使用 `increment_traj` 积累样本
   - 达到 batch_size 后触发 LLM 总结
   - 总结后清空增量区，等待下一批

---

## 核心差异总结表

| 维度 | ShortMemory | LocalMemory | GlobalMemory |
|------|-------------|-------------|--------------|
| **时间尺度** | 单步交互 | 单次 trial | 跨 trial 持久化 |
| **空间范围** | 当前轨迹 | 单环境隔离 | 全局共享 |
| **存储内容** | action-observation 对 | 专家反思记录 | 环境知识 + 任务经验 |
| **更新频率** | 每步更新 | 每轨迹更新 | 批量总结更新 |
| **检索方式** | 顺序召回 | 按 env_idx 召回 | 基于描述/类型检索 |
| **是否向量化** | 否 | 否 | 可选（ChromaDB） |
| **Prompt 作用** | 提供当前上下文 | 提供历史反思参考 | 提供先验知识指导 |

这三种记忆系统模拟了人类的认知机制：**短时记忆**处理当前任务，**本地记忆**积累经验教训，**全局记忆**提炼通用知识，三者协同工作实现持续学习和知识迁移。