import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from .cdmem import CDMemAgent


# system prompt 只保留稳定身份；具体任务规则放到 inference prompt 中。
SOLVER_SYSTEM_PROMPT = """
You are a smart agent designed to solve problems.
"""

GROUND_TRUTH_SYSTEM_PROMPT = """
You are the ground-truth correction agent in a multi-agent ALFWorld system.
When invoked, provide concise corrective help for the current task state.
"""

TASK_DEFINITION_PROMPT = """
You are now in a household environment called Alfworld, and your tasks include locating objects, heating or cooling items, and other similar activities.

NOTE:
- You must strictly follow the syntactic structure of the steps
    - think: <brief private reasoning>
    - look
    - inventory
    - go to (receptacle)
    - open (receptacle)
    - close (receptacle)
    - take (object) from (receptacle)
    - move (object) to (receptacle)
    - put (object) in/on (receptacle)
    - examine (something)
    - use (object)
    - heat (object) with (receptacle)
    - clean (object) with (receptacle)
    - cool (object) with (receptacle)
    - slice (object) with (object)

- You must check carefully whether your output command is consistent with the allowed commands above!!! Any output that is not among the commands listed above is not permitted!!!
"""


@dataclass
class CDMemAutoGenMember:
    """
    一个轻量版的 autogen agent。

    它只保留当前任务真正需要的字段：
    - name/role: 对齐 autogen 里的 Agent 信息
    - local_memory/global_memory: agent 私有的 CDMem memory
    - logging_dir: agent 自己的 memory 日志目录

    short memory 不放在这里，因为 ALFWorld 是单环境实时交互，
    solver 和 ground_truth 必须看到同一条 action-observation 轨迹。
    """

    name: str
    role: str
    local_memory: object
    global_memory: object
    logging_dir: str

    def response(
        self,
        llm,
        user_prompt: str,
        stop=None,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> str:
        # 仍然使用当前 CDMem 项目的 llm_wrapper，不走 mas.llm。
        # system prompt 由调用方按阶段显式传入；
        # expert/reflection/summary 等 memory 更新阶段保持原 CDMem 的中性上下文。
        kwargs = dict(stop=stop, max_tokens=max_tokens)
        if system_prompt:
            kwargs["sys_msg"] = system_prompt
        return llm(user_prompt, **kwargs)


class CDMemAutoGenTeam:
    """
    轻量版 MetaMAS 容器。

    只实现 autogen.py 里本任务需要的 hire/get_agent 能力。
    """

    def __init__(self) -> None:
        self.agents_team: Dict[str, CDMemAutoGenMember] = {}

    def hire(self, agents: List[CDMemAutoGenMember]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent

    def get_agent(self, agent_name: str) -> Optional[CDMemAutoGenMember]:
        return self.agents_team.get(agent_name)

    def names(self) -> List[str]:
        return list(self.agents_team.keys())


class CDMemAutoGenAgent(CDMemAgent):
    """
    CDMem + AutoGen 风格调度的 ALFWorld agent。

    设计约束：
    - env / prompt_builder / fewshot_builder / memory 类沿用原始 CDMem
    - 调度结构对齐 autogen.py: solver 默认行动，ground_truth 只在 solver 卡住时介入
    - short memory 共享，表示当前 episode 的实时轨迹
    - local/global memory 按 agent 独立
    - canonical local/global memory 仍保留，用于兼容原 CDMem logger 和汇总日志
    """

    def __init__(
        self,
        num_trials,
        num_envs,
        max_steps,
        logging_dir,
        env,
        llm_wrapper,
        model,
        start_trial_num,
        short_memory,
        local_memory,
        global_memory,
        prompt_builder,
        fewshot_builder,
        *args,
        **kwargs,
    ):
        # 这里继承父类的属性
        super().__init__(
            num_trials,
            num_envs,
            max_steps,
            logging_dir,
            env,
            llm_wrapper,
            model,
            start_trial_num,
            short_memory,
            local_memory,
            global_memory,
            prompt_builder,
            fewshot_builder,
            *args,
            **kwargs,
        )
        self.local_memory_cls = local_memory  # 每个agent独立的local memory
        # self.global_memory_cls = global_memory

        # agent_memory_root 下保存每个 agent 自己的 local memory 日志。
        # 例如：
        # logs/.../agent_memory/solver/local_memory_trial_0.json
        self.agent_memory_root = os.path.join(self.logging_dir, "agent_memory")
        os.makedirs(self.agent_memory_root, exist_ok=True)

        # 仅用于 trial log 展示：记录每个 action 是哪个 agent 输出的。
        # 真正给 LLM 的 short memory 仍然是 self.short_memory，保持原 CDMem 格式。
        self.agent_short_history = []

        # 记录真正参与trajectory的agent name
        self.trajectory_agent_names = set()

        # agent团队构建
        self.team = self._build_team()

    def _build_team(self) -> CDMemAutoGenTeam:
        """
        构建 autogen 风格的双 agent 团队。

        solver:
            默认每一步都由它给 action。
        ground_truth:
            只有当 solver 连续重复同一个 action、被判定 stuck 时才介入。
        """
        role_specs = [
            (
                "solver",
                "solver",
            ),
            (
                "ground_truth",
                "ground truth agent",
            ),
        ]
        team = CDMemAutoGenTeam()
        members = []
        for name, role in role_specs:
            agent_logging_dir = os.path.join(self.agent_memory_root, name)
            os.makedirs(agent_logging_dir, exist_ok=True)

            # global memory 团队共用同一份 self.global_memory。
            # global memory 记录的是 env known_obs / task action_guidance 这类
            # 客观的高层指导，属于环境属性而非 agent 私有视角；而且 solver 和
            # ground_truth 在同一环境里共享同一条 episode，所以 ground_truth 的
            # 纠正经验也应该能被 solver 后续召回。
            members.append(
                CDMemAutoGenMember(
                    name=name,
                    role=role,
                    # local_memory 仍每个 agent 一份，因为 reflection 是第一人称
                    # 的学习链，solver/ground_truth 视角不同，分开更合理。
                    local_memory=self.local_memory_cls(self.num_envs),
                    global_memory=self.global_memory,  # 继承自父类
                    logging_dir=agent_logging_dir,
                )
            )
        # 组装agents到team
        team.hire(
            members
        )
        return team

    def build_infer_system_prompt(self, agent_name: str) -> str:
        """
        只在推理阶段按需构建 system prompt。

        agent 对象不持有固定 system prompt，避免 expert/reflection/summary
        等 memory 更新阶段误用角色提示。后续如果要继续细分不同阶段的
        system prompt，也只需要改这里或新增对应 builder。
        """
        if agent_name == "ground_truth":
            return (GROUND_TRUTH_SYSTEM_PROMPT + "\n" + TASK_DEFINITION_PROMPT).strip()
        return (SOLVER_SYSTEM_PROMPT + "\n" + TASK_DEFINITION_PROMPT).strip()

    def run_trajectory(self, env_idx, init_ob, to_print=True):
        """
        跑一个 ALFWorld episode。

        调度方式对齐 autogen.py:
        1. solver 先根据 CDMem prompt 输出 action
        2. 如果 solver 连续重复同一动作，则调用 ground_truth 重新输出 action
        3. 被选中的 action 执行到环境中
        4. action/observation 写入共享 short memory

        返回三项：
        - log_history_log: 写 trial log 的版本，action 前带 [solver]/[ground_truth]
        - memory_history_log: 原始 CDMem 格式，用于 expert/reflection/memory 更新
        - is_success: 当前 episode 是否成功
        """
        cur_step = 0
        turn_idx = 0
        action_history: List[str] = []
        active_agent_name = "solver"

        print(init_ob)

        # short memory 是当前 episode 的共享轨迹，每个新环境都要清空。
        self.short_memory.reset()

        # agent_short_history 只服务于日志显示，不参与 LLM prompt。
        self.agent_short_history = []
        self.trajectory_agent_names = set()

        # think 不计入 cur_step。为了防止模型一直输出 think 导致无限循环，
        # 额外用 turn_idx 限制总 LLM 调用轮数。
        max_turns = self.max_steps * 2
        while cur_step < self.max_steps and turn_idx < max_turns:
            # 默认由 solver 行动。
            solver = self.team.get_agent("solver")
            active_agent = solver
            active_agent_name = "solver"
            self.trajectory_agent_names.add(active_agent_name)
            infer_prompt = self.build_infer_prompt(
                env_idx, init_ob, active_agent_name)

            action = active_agent.response(
                self.llm,
                infer_prompt,
                stop=["\n"],
                system_prompt=self.build_infer_system_prompt(
                    active_agent_name),
            ).strip()
            # 格式化为规范化action
            action = self.env.action_parser(action)
            action = self.normalize_place_action(action)
            if not action:
                action = "think: I need to choose a valid next action."

            # autogen.py 的核心切换逻辑：
            # 如果 solver 的当前 action 和前两步完全一样，认为它陷入重复循环，
            # 这一步改由 ground_truth 给出替代 action。
            if self._solver_stuck(action, action_history):
                active_agent_name = "ground_truth"
                self.trajectory_agent_names.add(active_agent_name)
                active_agent = self.team.get_agent(active_agent_name)
                stuck_context = self.build_stuck_context(
                    action, action_history)
                infer_prompt = self.build_infer_prompt(
                    env_idx,
                    init_ob,
                    active_agent_name,
                    stuck_context=stuck_context,
                )
                action = active_agent.response(
                    self.llm,
                    infer_prompt,
                    stop=["\n"],
                    system_prompt=self.build_infer_system_prompt(
                        active_agent_name),
                ).strip()
                action = self.env.action_parser(action)
                action = self.normalize_place_action(action)
                if not action:
                    action = "think: I need to choose a valid next action."

            # 共享 short memory 仍保持原始 CDMem 的 action/observation 结构；
            # 同时 agent_short_history 额外记录 action 来自哪个 agent，供日志展示。
            self._add_short_memory("action", action, active_agent_name)
            observation, reward, done, exhausted, info = self.env.step(action)
            observation_text = "Observation: " + observation
            self._add_short_memory(
                "observation", observation_text, active_agent_name)

            if to_print:
                print(f"[{active_agent_name}] Action: {action}")
                print(f"Observation: {observation}")
                sys.stdout.flush()

            action_history.append(action)
            turn_idx += 1

            # done/exhausted 时构造两份 history：
            # - memory_history_log: 原始格式，用于后续 memory 更新
            # - log_history_log: 带 agent 名，用于 trial_*.log 可读性
            if done:
                memory_history_log = self.build_infer_prompt(
                    env_idx, init_ob, active_agent_name)
                log_history_log = self.build_labeled_history_log(
                    memory_history_log)
                return log_history_log, memory_history_log, True
            # * 暂时用不上，因为会和MAS冲突
            elif exhausted:
                memory_history_log = self.build_infer_prompt(
                    env_idx, init_ob, active_agent_name)
                log_history_log = self.build_labeled_history_log(
                    memory_history_log)
                return log_history_log, memory_history_log, False
            # 如果是think action，不计入cur_step
            if action.startswith("think:"):
                continue
            cur_step += 1

        # 达到 max_steps 或 max_turns 也算失败返回。
        memory_history_log = self.build_infer_prompt(
            env_idx, init_ob, active_agent_name)
        log_history_log = self.build_labeled_history_log(memory_history_log)
        # log_history_log是带agent name的，用于trial log展示；
        # memory_history_log是原始格式，用于后续 memory 更新
        return log_history_log, memory_history_log, False

    def _solver_stuck(self, current_action: str, action_history: List[str]) -> bool:
        """判断 solver 是否陷入重复动作循环。"""
        return (
            len(action_history) >= 2
            and current_action == action_history[-1]
            and current_action == action_history[-2]
        )

    # * 暂时不用
    def build_stuck_context(self, current_action: str, action_history: List[str]) -> str:
        recent_actions = action_history[-3:] + [current_action]
        recent_action_text = "\n".join(
            f"{idx + 1}. {action}" for idx, action in enumerate(recent_actions)
        )
        return f"""The solver is about to repeat an action that already appears in the recent action history.

Recent actions:
{recent_action_text}

Stuck action to avoid:
{current_action}"""

    # 更新short memory（trajectory）
    def _add_short_memory(self, label: str, value: str, active_agent_name: str) -> None:
        """
        同时维护两份短轨迹：
        - self.short_memory: 原始 CDMem prompt 使用，不带 agent 名
        - self.agent_short_history: trial log 使用，带 agent 名
        """
        self.short_memory.add(label, value)
        self.agent_short_history.append(
            dict(label=label, value=value, agent_name=active_agent_name)
        )

    # 仅用于日志，非真实memory
    def build_labeled_history_log(self, memory_history_log: str) -> str:
        """
        把原始 CDMem prompt 中的 Task Interactive Trajectory 替换为带 agent 名的版本。

        这样 trial_*.log 可读性更好，同时不会影响 memory_history_log，
        后者仍以原始格式进入 expert/reflection/global memory 流程。
        """
        marker = "## Task Interactive Trajectory:"
        if marker not in memory_history_log:
            return memory_history_log
        prefix = memory_history_log.split(marker, 1)[0]
        return f"{prefix}{marker}\n{self.recall_labeled_short_memory()}"

    def recall_labeled_short_memory(self) -> str:
        """生成带 agent 名的短期轨迹文本。"""
        s = "\n"
        for i, item in enumerate(self.agent_short_history):
            if item["label"] == "action":
                s += f'> [{item["agent_name"]}] {item["value"]}'
            elif item["label"] == "observation":
                s += item["value"]
            if i != len(self.agent_short_history) - 1:
                s += "\n"
        return s

    def build_infer_prompt(
        self,
        env_idx,
        init_ob,
        agent_name: Optional[str] = None,
        stuck_context: Optional[str] = None,
    ):
        """
        构建当前 agent 的推理 prompt。

        solver 仍使用原始 CDMem inference prompt。
        ground_truth 使用 CDMemPromptBuilder 新增的 get_ground_truth_inference_prompts。

        两者 memory 来源都按 agent 区分：
        - short memory: 团队共享 self.short_memory
        - local memory: 当前 agent 私有 local_memory
        - global memory: 当前 agent 私有 global_memory

        fewshot builder 仍然用 self.logging_dir 读取 trial_*.log，因为完整轨迹
        只写在主 trial log 中；agent.logging_dir 只保存各自的 memory json。
        """
        short_memories = self.short_memory.recall()
        local_memories = self._recall_agent_local_memories(agent_name, env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]

        env_description, task_description = self.process_before_infer(init_ob)
        agent_global_memory = self._get_agent_global_memory(agent_name)
        known_obs_history, action_guidance_history = agent_global_memory.recall(
            env_description, task_description
        )
        fewshots = self.fewshot_builder.get_inference_fewshots(
            self.env.name,
            env_description,
            task_description,
            agent_global_memory,
            self.logging_dir,
        )
        if agent_name == "ground_truth" and hasattr(
            self.prompt_builder, "get_ground_truth_inference_prompts"
        ):
            return self.prompt_builder.get_ground_truth_inference_prompts(
                init_ob,
                fewshots,
                local_memories,
                short_memories,
                known_obs_history,
                action_guidance_history,
                stuck_context=stuck_context,
            )

        return self.prompt_builder.get_inference_prompts(
            init_ob,
            fewshots,
            local_memories,
            short_memories,
            known_obs_history,
            action_guidance_history,
        )

    # * local memory只根据环境编号来召回
    def _recall_agent_local_memories(self, agent_name: Optional[str], env_idx: int):
        """读取当前 agent 在该 env 上的 local reflections。"""
        if agent_name:
            agent = self.team.get_agent(agent_name)
            if agent is not None:
                return agent.local_memory.recall(env_idx)
        return self.local_memory.recall(env_idx)

    def _get_agent_global_memory(self, agent_name: Optional[str] = None):
        """获取 global memory；团队共用同一份 self.global_memory。"""
        return self.global_memory

    def update_local_memory(self, history_log, is_success, env_idx):
        """
        更新 local memory。

        这里分两层：
        1. 每个 agent 各自生成 reflection，写入自己的 local_memory
        2. 再把所有 agent reflection 合并成 canonical reflection，写入 self.local_memory

        canonical local memory 主要为了兼容原 CDMem logger/global summary 文件格式；
        真正 agent 推理时读取的是各自的 local_memory。
        """
        if self.local_memory.is_skip(env_idx):
            return self._empty_expert_trajectory(history_log, is_success)

        agent_trajectories = {}
        combined_reflections = []
        for agent_name, agent in self.team.agents_team.items():
            # 仅每个参与过本条 trajectory 的 agent 都单独做 expert encoding，
            # if agent_name not in self.trajectory_agent_names:
            #     continue

            # 但 memory 更新阶段不注入 agent system prompt，避免角色设定污染
            # 原始 CDMem 的 expert/reflection 记忆维护流程。
            expert_prompt = self.build_expert_prompt_for_agent(
                history_log,
                agent_name,
            )
            expert_result = agent.response(
                self.llm,
                expert_prompt,
                max_tokens=512,
            )
            print(f"[DEBUG] {agent_name} expert_result: {expert_result}")

            # 每个 agent 用自己的 expert_result 和 local_memory 参与 reflection prompt，
            # 所以 solver/ground_truth 会形成不同的历史反思链。
            reflection_prompt = self.build_reflection_prompt_for_agent(
                history_log,
                is_success,
                expert_result,
                env_idx,
                agent_name,
            )
            reflection_result = agent.response(
                self.llm,
                reflection_prompt,
                max_tokens=512,
            )
            print(
                f"[DEBUG] {agent_name} reflection_result: {reflection_result}")
            expert_trajectory = self.process_after_reflection(
                expert_result,
                reflection_result,
                history_log,
                is_success,
            )
            # 写入 agent 私有 local memory。
            agent.local_memory.add(env_idx, expert_trajectory)
            agent_trajectories[agent_name] = expert_trajectory
            combined_reflections.append(
                f"{agent_name} ({agent.role}): {reflection_result.strip()}"
            )

        # canonical trajectory 用于主 local_memory_trial_*.json 和主 global memory。
        # 它只是兼容原 CDMem 单 agent 日志/检索格式的团队汇总层；
        # 真正 per-agent memory 已经写入 agent.local_memory。
        canonical_trajectory = self._build_canonical_trajectory(
            agent_trajectories,
            combined_reflections,
            history_log,
            is_success,
        )
        canonical_trajectory["agent_reflections"] = agent_trajectories
        self.local_memory.add(env_idx, canonical_trajectory)
        print(
            f"[DEBUG] canonical local memory={self.local_memory.recall(env_idx)}")
        return canonical_trajectory

    # TODO 可能考虑删掉
    def _build_canonical_trajectory(
        self,
        agent_trajectories,
        combined_reflections,
        history_log,
        is_success,
    ):
        """
        从 per-agent expert trajectories 生成一份团队级 canonical trajectory。

        优先沿用 solver 的 env/task/location/function/action 字段，因为 solver 是
        默认执行 agent，也最接近原始 CDMem 的单 agent 行为；reflection 字段则保存
        所有参与 agent 的反思，方便主日志和主 global memory 仍能看到团队视角。
        """
        canonical_trajectory = (
            agent_trajectories.get("solver")
            or next(iter(agent_trajectories.values()), None)
        )
        if canonical_trajectory is None:
            canonical_trajectory = self._empty_expert_trajectory(
                history_log,
                is_success,
            )
        else:
            canonical_trajectory = dict(canonical_trajectory)
        canonical_trajectory["reflection"] = "\n".join(combined_reflections)
        return canonical_trajectory

    # 分agent的expert prompt
    def build_expert_prompt_for_agent(self, history_log, agent_name):
        """构建某个 agent 专属的 expert encoding prompt。"""
        if agent_name == "ground_truth" and hasattr(
            self.prompt_builder,
            "get_ground_truth_expert_prompts",
        ):
            fewshots = self.fewshot_builder.get_ground_truth_expert_fewshots()
            return self.prompt_builder.get_ground_truth_expert_prompts(
                history_log,
                fewshots,
            )
        return self.build_expert_prompt(history_log)

    # 分agent的reflection prompt
    def build_reflection_prompt_for_agent(
        self,
        history_log,
        is_success,
        expert_result,
        env_idx,
        agent_name,
    ):
        """构建某个 agent 专属的 reflection prompt。"""
        # 根据env编号来召回
        local_memories = self._recall_agent_local_memories(agent_name, env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]  # 最多只召回3条
        if agent_name == "ground_truth" and hasattr(
            self.prompt_builder,
            "get_ground_truth_reflection_prompts",
        ):
            fewshots = self.fewshot_builder.get_ground_truth_reflection_fewshots(
                is_success,
            )
            return self.prompt_builder.get_ground_truth_reflection_prompts(
                history_log,
                is_success,
                fewshots,
                local_memories,
                expert_result,
            )
        fewshots = self.fewshot_builder.get_reflection_fewshots(is_success)
        return self.prompt_builder.get_reflection_prompts(
            history_log,
            is_success,
            fewshots,
            local_memories,
            expert_result,
        )

    def _empty_expert_trajectory(self, history_log, is_success):
        """skip 分支的兜底返回，保持 expert_trajectory 字段结构完整。"""
        env_description = task_description = ""
        scenario = history_log.split("Here is the task:")[-1].strip()
        for line in scenario.splitlines():
            if line.startswith("You are in the middle of a room."):
                env_description = line.strip()
            elif line.startswith("Your task is to:"):
                task_description = line.split(
                    "Your task is to:", 1)[-1].strip()
        return dict(
            env=env_description,
            task=task_description,
            location="",
            function="",
            action="",
            reflection="",
            is_success=is_success,
        )

    def log_agent_local_memory(self, trial_idx: int) -> None:
        """把每个 agent 的 local memory 写到自己的目录。"""
        import json

        for agent_name, agent in self.team.agents_team.items():
            path = os.path.join(
                agent.logging_dir,
                f"local_memory_trial_{trial_idx}.json",
            )
            with open(path, "w") as wf:
                json.dump(agent.local_memory.history, wf, indent=4)

    def run(self):
        """
        主运行循环。

        结构保留原始 CDMem：
        trial -> env -> trajectory -> local memory -> global memory -> log。

        额外增加：
        - per-agent local memory log
        - per-agent global memory update/log
        - trial log 使用带 agent 名称的 history
        """
        for trial_idx in range(self.start_trial_num, self.num_trials):
            self.logger.log_world_start(trial_idx)
            num_successes = 0
            num_additional_successes = 0

            for env_idx in range(self.num_envs):
                init_ob, info = self.env.reset()
                print(f"{env_idx} using {self.env.name}")
                if self.local_memory.is_success(env_idx):
                    # canonical memory 里已经成功的环境沿用原 CDMem 行为：跳过。
                    num_successes += 1
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.logger.log_trial_success(trial_idx, env_idx)
                    continue

                history_log, memory_history_log, is_success = self.run_trajectory(
                    env_idx, init_ob)
                if is_success:
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.local_memory.set_success(env_idx)
                    for agent in self.team.agents_team.values():
                        agent.local_memory.set_success(env_idx)
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    self.logger.log_world_fail(trial_idx, env_idx)

                self.logger.log_trial_content(
                    # 写日志用带 [solver]/[ground_truth] 的版本。
                    history_log,
                    is_success,
                    trial_idx,
                    env_idx,
                )
                expert_trajectory = self.update_local_memory(
                    # 更新 memory 用原始 CDMem 格式，避免 agent 标签干扰解析和总结。
                    memory_history_log,
                    is_success,
                    env_idx,
                )
                # 先写 local memory，因为 GlobalMemory.short2long 会按 trial/env 索引
                # 读取 local_memory_trial_*.json 作为样本。
                self.logger.log_local_memory(trial_idx)
                self.log_agent_local_memory(trial_idx)

                # global memory 现在团队共用一份，只需要更新和日志一次。
                self.update_global_memory(
                    expert_trajectory, env_idx, trial_idx)
                self.logger.log_global_memory(trial_idx)

            self.env.close()
            self.logger.log_world_end(
                trial_idx,
                num_successes,
                num_additional_successes,
            )
            self.env.reload()

    # * 暂时用不上
    def _action_system_message(self) -> str:
        """
        可选的 ALFWorld action 格式约束 system prompt。

        当前 _build_team() 里 action_contract 设为空，表示先保持 autogen 原始
        system prompt 设定。如果后续希望强约束动作格式，可以把 _build_team()
        中的 action_contract 改回 self._action_system_message()。
        """
        return """
You are controlling an ALFWorld TextWorld agent. You need to output your thinking/reason/plan to solve the task, and select a correct action to execute.
At each turn, output exactly ONE line and do not output observations, explanations, markdown, or multiple actions.

Available action templates:
- think: <brief private reasoning>
- look: look around your current location
- inventory: check your current inventory
- go to (receptacle): move to a receptacle
- open (receptacle): open a receptacle
- close (receptacle): close a receptacle
- take (object) from (receptacle): take an object from a receptacle
- move (object) to (receptacle): place an object in or on a receptacle
- put (object) in/on (receptacle): place an object in or on a receptacle
- examine (something): examine a receptacle or an object
- use (object): use an object
- heat (object) with (receptacle): heat an object using a receptacle
- clean (object) with (receptacle): clean an object using a receptacle
- cool (object) with (receptacle): cool an object using a receptacle
- slice (object) with (object): slice an object using a sharp object

Note: Use exact object and receptacle names with their numbers from observations, such as "apple 1", "drawer 2", or "sinkbasin 1". Do not invent object names, receptacle names, ids, or environment feedback.
Please note, the task interactive trajectory is realtime feedback from environment. You are required to interact with the environment to complete the task.
So, you need to output your thinking or a valid action, and the action will be executed in the environment.
- If you output your thinking, the environment will simple respond with "OK".
- If you output a valid action, the environment will return a new observation.
- If you meet operation failure, the environment will return "Nothing happens", which means the current observation doesn't match the current action. Please rethink and choose a different action.
""".strip()
