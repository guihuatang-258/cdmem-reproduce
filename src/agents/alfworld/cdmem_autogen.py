import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .cdmem import CDMemAgent
from .cdmem_autogen_system_prompts import CDMEM_AUTOGEN_PROMPT


@dataclass
class CDMemAutoGenMember:
    """
    一个轻量版的 autogen agent。

    它只保留当前任务真正需要的字段：
    - name/role: 对齐 autogen 里的 Agent 信息
    - local_memory: agent 私有的 local memory（reflection 链）
    - global_memory: 团队共用的 global memory（指向同一实例）
    - logging_dir: agent 自己的 memory 日志目录

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
        max_retries: int = 3,
    ) -> str:
        # 仍然使用当前 CDMem 项目的 llm_wrapper，不走 mas.llm。
        # system prompt 由调用方按阶段显式传入；
        # expert/reflection/summary 等 memory 更新阶段保持原 CDMem 的中性上下文。
        kwargs = dict(stop=stop, max_tokens=max_tokens)
        if system_prompt:
            kwargs["sys_msg"] = system_prompt

        for attempt in range(max_retries):
            try:
                result = llm(user_prompt, **kwargs)
                if result and len(result.strip()) >= 1:
                    return result
                print(
                    f"⚠️  [{self.name}] empty response, "
                    f"retry {attempt + 1}/{max_retries}"
                )
            except (SystemExit, Exception) as e:
                # llm_wrapper 在 API 层重试耗尽后会 sys.exit(1)，
                # 这里拦截 SystemExit/Exception，避免进程直接终止。
                print(
                    f"⚠️  [{self.name}] LLM call error, "
                    f"retry {attempt + 1}/{max_retries}: {e}"
                )

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s → 2s → 4s ...

        print(f"❌ [{self.name}] all {max_retries} retries exhausted")
        return ""


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
    - short memory 团队共享，表示当前 episode 的实时轨迹
    - local memory 按 agent 独立（reflection 是第一人称学习链，视角不同）
    - global memory 团队共用（env known_obs / task action_guidance 是客观的高层指导）
    - canonical local memory 仍保留，用于兼容原 CDMem logger 和 global summary 格式
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
        self.local_memory_cls = local_memory  # 用于给每个 agent new 独立的 local memory
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

    def build_infer_system_prompt(self, agent_name: str, stuck_reason: str = "") -> str:
        """
        只在推理阶段按需构建 system prompt。

        agent 对象不持有固定 system prompt，避免 expert/reflection/summary
        等 memory 更新阶段误用角色提示。后续如果要继续细分不同阶段的
        system prompt，也只需要改这里或新增对应 builder。
        """
        if agent_name == "ground_truth":
            return (CDMEM_AUTOGEN_PROMPT.ground_truth_system_prompt + "\n" + CDMEM_AUTOGEN_PROMPT.task_definition_prompt_ground_truth).strip()
        return (CDMEM_AUTOGEN_PROMPT.solver_system_prompt + "\n" + CDMEM_AUTOGEN_PROMPT.task_definition_prompt_solver).strip()

    def run_trajectory(self, env_idx, init_ob, to_print=True):
        """
        跑一个 ALFWorld episode。

        调度方式对齐 autogen.py:
        1. solver 先根据 CDMem prompt 输出 action
        2. 如果 solver 卡住了，则调用 ground_truth 重新输出 action
        3. 被选中的 action 执行到环境中
        4. action/observation 写入共享 short memory

        返回三项：
        - log_history_log: 写 trial log 的版本，action 前带 [solver]/[ground_truth]
        - memory_history_log: 原始 CDMem 格式，用于 expert/reflection/memory 更新
        - is_success: 当前 episode 是否成功
        """
        cur_step = 0
        turn_idx = 0
        action: str = ""  # 当前 action，先声明避免 UnboundLocalError
        active_agent_name = "solver"

        # ground_truth 介入后继续保持，避免 solver 马上又陷入 think loop
        # 形成 "solver think → ground_truth 覆盖 → solver think → ..." 的死循环
        # !暂时不启用
        keep_ground_truth = False

        print(init_ob)

        # short memory 是当前 episode 的共享轨迹，每个新环境都要清空。
        self.short_memory.reset()

        # agent_short_history 只服务于日志显示，不参与 LLM prompt。
        self.agent_short_history = []
        self.trajectory_agent_names = set()

        # think 不计入 cur_step。为了防止模型一直输出 think 导致无限循环，
        # 额外用 turn_idx 限制总 LLM 调用轮数。
        max_turns = self.max_steps * 3
        while cur_step < self.max_steps and turn_idx < max_turns:
            # ── 决定谁来行动 ──
            # keep_ground_truth: 上轮 ground_truth 介入后仍没打破僵局（think 或
            # Nothing happens），继续用 ground_truth，避免 solver 马上又陷入 think loop
            is_stuck_1, stuck_reason = self._solver_stuck_1()
            use_ground_truth = is_stuck_1 or keep_ground_truth

            if use_ground_truth:
                active_agent_name = "ground_truth"
                self.trajectory_agent_names.add(active_agent_name)
                active_agent = self.team.get_agent(active_agent_name)
                stuck_context = self.build_stuck_context(action)
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
            else:
                # 由 solver 行动。
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

            # ── solver 路径的兜底检测 ──
            # 只有在 keep_ground_truth=False 时才需要 _solver_stuck_2，
            # 因为 keep_ground_truth=True 时已经直接用 ground_truth 了
            if not keep_ground_truth:
                is_stuck_2, stuck_reason = self._solver_stuck_2(
                    action, active_agent_name)
                if is_stuck_2 and not is_stuck_1:
                    active_agent_name = "ground_truth"
                    self.trajectory_agent_names.add(active_agent_name)
                    active_agent = self.team.get_agent(active_agent_name)
                    stuck_context = self.build_stuck_context(action)
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
            # 这里更新了agent_short_history
            self._add_short_memory("action", action, active_agent_name)
            observation, reward, done, exhausted, info = self.env.step(action)
            observation_text = "Observation: " + observation
            self._add_short_memory(
                "observation", observation_text, active_agent_name)

            if to_print:
                print(f"🏃[{active_agent_name}] Action: {action}")
                print(f"🌍Observation: {observation}")
                sys.stdout.flush()

            turn_idx += 1

            # ── 决定下轮是否继续用 ground_truth ──
            # 如果 ground_truth 当前这步也没打破僵局（Nothing happens），
            # 下轮继续用 ground_truth，避免还给 solver 后又陷入 think loop。
            # ?如果 ground_truth 自己也 stuck 了怎么办
            # if active_agent_name == "ground_truth":
            #     # is_gt_stuck, _ = self._ground_truth_stuck()
            #     # if is_gt_stuck:
            #     #     keep_ground_truth = False
            #     # else:
            #     keep_ground_truth = "Nothing happens" in observation
            # else:
            #     keep_ground_truth = False

            # done/exhausted 时构造两份 history：
            # - memory_history_log: 原始格式，用于后续 memory 更新
            # - log_history_log: 带 agent 名，用于 trial_*.log 可读性
            if done:
                memory_history_log = self.build_infer_prompt(
                    env_idx, init_ob, "solver")
                log_history_log = self.build_labeled_history_log(
                    memory_history_log)
                return log_history_log, memory_history_log, True
            # * 暂时用不上，因为会和MAS冲突
            elif exhausted:
                memory_history_log = self.build_infer_prompt(
                    env_idx, init_ob, "solver")
                log_history_log = self.build_labeled_history_log(
                    memory_history_log)
                return log_history_log, memory_history_log, False
            # 如果是think action，不计入cur_step
            if action.startswith("think:"):
                continue
            cur_step += 1

        # 达到 max_steps 或 max_turns 也算失败返回。
        # memory_history_log 始终用 solver 格式（原始 CDMem），供后续 memory 更新使用。
        memory_history_log = self.build_infer_prompt(
            env_idx, init_ob, "solver")
        log_history_log = self.build_labeled_history_log(memory_history_log)
        # log_history_log是带agent name的，用于trial log展示；
        # memory_history_log是原始格式，用于后续 memory 更新
        return log_history_log, memory_history_log, False

    def _solver_stuck_1(self) -> tuple:
        """检测连续 "Nothing happens"（真实时间线，不按 agent 过滤）。

        避免 ground_truth 介入后 solver 的旧 "Nothing happens" 持续触发误判：
          [solver: Nothing][solver: Nothing][ground_truth: arrives]...[solver: OK]
          旧: solver_obs[-2:]=[Nothing,Nothing] → 误判
          新: all_obs[-2:]=[arrives, OK] → 不触发
        """
        all_obs = [item["value"] for item in self.agent_short_history
                   if item["label"] == "observation"]
        if (len(all_obs) >= 2
                and "Nothing happens" in all_obs[-1]
                and "Nothing happens" in all_obs[-2]):
            return True, "nothing happens"
        return False, ""

    def _solver_stuck_2(
        self,
        current_action: str,
        current_agent_name: str = "",
    ) -> tuple:
        """
        判断当前 agent 是否陷入僵局，需要 ground_truth 介入。

        触发条件（满足任一即介入）：
        1. 同 agent 重复：当前 agent 的 action 与自己上一次的 action 完全相同
        2. 跨步重复：当前 action 与上一步 action 完全相同（无论 agent 来源）
        3. 连续空想：当前及前两步（真实时间线，不按 agent 过滤）都是 think
        """
        all_actions = [
            h for h in self.agent_short_history if h["label"] == "action"
        ]

        # 条件 1：同一 agent 重复自己上一次的 action
        # （例如 ground_truth 连续两次给出相同纠正也视为 stuck）
        # same_agent_actions = [
        #     h["value"] for h in all_actions
        #     if h["agent_name"] == current_agent_name
        # ]
        # if (len(same_agent_actions) >= 1
        #         and current_action == same_agent_actions[-1]):
        #     return True, "repeat_same_agent"

        # 条件 2：当前 action 与上一步完全相同（无论 agent 来源）
        if (len(all_actions) >= 1
                and current_action == all_actions[-1]["value"]):
            return True, "repeat"

        # 条件 3：连续3次 think 检测（无论 agent 来源）。
        recent_actions = [a["value"] for a in all_actions[-2:]]
        if (current_action.startswith("think:")
                and len(recent_actions) >= 2
                and all(a.startswith("think:") for a in recent_actions[-2:])):
            return True, "think"

        return False, ""

    # * 暂时不用
    def _ground_truth_stuck(self) -> tuple:
        """
        判断 ground_truth 自身是否也陷入僵局（需要交还 solver）。

        注意：调用时 agent_short_history 已包含当前 action/observation，
        所以 gt_actions[-1] 即当前 action，重复检测要跟 [-2] 比。

        触发条件（满足任一即 stuck）：
        1. 重复自己：当前 action == ground_truth 上一次 action
        2. 违规 think：ground_truth 输出了 think（system prompt 已禁止）
        3. 连续 Nothing happens：ground_truth 最近两步 obs 都是 Nothing happens
        """
        gt_actions = [h["value"] for h in self.agent_short_history
                      if h["label"] == "action"
                      and h["agent_name"] == "ground_truth"]
        gt_obs = [h["value"] for h in self.agent_short_history
                  if h["label"] == "observation"
                  and h["agent_name"] == "ground_truth"]

        # 条件 1：ground_truth 重复自己上一次的 action
        if len(gt_actions) >= 2 and gt_actions[-1] == gt_actions[-2]:
            return True, "gt_repeat_self"

        # 条件 2：ground_truth 输出了 think（本不该输出）
        if len(gt_actions) >= 1 and gt_actions[-1].startswith("think:"):
            return True, "gt_think"

        # 条件 3：ground_truth 连续两次 "Nothing happens"
        if (len(gt_obs) >= 2
                and "Nothing happens" in gt_obs[-1]
                and "Nothing happens" in gt_obs[-2]):
            return True, "gt_nothing_happens"

        return False, ""

    # 详细说明stuck的上下文
    def build_stuck_context(
        self,
        current_action: str,
    ) -> str:
        all_actions = [
            h for h in self.agent_short_history if h["label"] == "action"
        ]
        recent_actions = all_actions[-3:] + [
            {"value": current_action, "agent_name": "（当前）"}]
        recent_action_text = "\n".join(
            f"{idx + 1}. [{h['agent_name']}] {h['value']}"
            for idx, h in enumerate(recent_actions)
        )
        return f"""
### Recent actions (with agent source):
{recent_action_text}

### Stuck action to avoid:
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

        memory 来源：
        - short memory: 团队共享 self.short_memory（当前 episode 的实时轨迹）
        - local memory: 当前 agent 私有 local_memory（按 env_idx 索引的 reflection 链）
        - global memory: 团队共用 self.global_memory（env known_obs + task action_guidance）

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

    def update_local_memory(self, history_log, memory_history_log, is_success, env_idx):
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
            if agent_name not in self.trajectory_agent_names:
                continue

            # memory 更新阶段不注入 agent system prompt，避免多余信息污染
            # 分agent生成expert result
            expert_prompt = self.build_expert_prompt_for_agent(
                history_log,
                memory_history_log,
                agent_name,
            )
            expert_result = agent.response(
                self.llm,
                expert_prompt,
                max_tokens=512,
            )
            print(f"⚠️[DEBUG] {agent_name} expert_result: {expert_result}")

            # 每个 agent 用自己的 expert_result 和自己的 local_memory 参与 reflection prompt，
            # 所以 solver/ground_truth 会形成不同的历史反思链。
            reflection_prompt = self.build_reflection_prompt_for_agent(
                history_log,
                memory_history_log,
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
                f"⚠️[DEBUG] {agent_name} reflection_result: {reflection_result}")

            # 解析为结构化expert_trajectory，并写入 agent 私有 local memory
            expert_trajectory = self.process_after_reflection(
                expert_result,
                reflection_result,
                history_log,
                is_success,
            )
            # 私有 local memory。
            agent.local_memory.add(env_idx, expert_trajectory)
            agent_trajectories[agent_name] = expert_trajectory

            # 在汇总的 local memory 中保存每个 agent 的reflection
            combined_reflections.append(
                f"({agent_name}): {reflection_result.strip()}"
            )

        # canonical trajectory 用于主 local_memory_trial_*.json 和共享 global memory。
        # 它的 env/task/location/function/action 取自 solver，
        # reflection 取两个 agent 合并版。
        # global memory 的 env 知识总结依赖其中的 function 字段，
        # task 知识总结依赖 action + reflection 字段。
        canonical_trajectory = self._build_canonical_trajectory(
            agent_trajectories,
            combined_reflections,
            history_log,
            is_success,
        )
        canonical_trajectory["agent_reflections"] = agent_trajectories
        # 兼容原结构，保存合并后agent的local memory
        self.local_memory.add(env_idx, canonical_trajectory)
        print(
            f"⚠️[DEBUG] canonical local memory={self.local_memory.recall(env_idx)}")
        return canonical_trajectory

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
    def build_expert_prompt_for_agent(self, history_log, memory_history_log, agent_name):
        """构建某个 agent 专属的 expert encoding prompt。

        - solver: 用 memory_history_log（无 agent 标签的原始 CDMem 格式）
        - ground_truth: 用 history_log（带 [solver]/[ground_truth] 标签的版本），
          这样纠正专家能看出哪些 action 是 solver 自己走的、哪些是 ground_truth 介入的。
        """
        if agent_name == "ground_truth" and hasattr(
            self.prompt_builder,
            "get_ground_truth_expert_prompts",
        ):
            fewshots = self.fewshot_builder.get_ground_truth_expert_fewshots()
            return self.prompt_builder.get_ground_truth_expert_prompts(
                history_log,
                fewshots,
            )
        return self.build_expert_prompt(memory_history_log)

    # 分agent的reflection prompt
    def build_reflection_prompt_for_agent(
        self,
        history_log,
        memory_history_log,
        is_success,
        expert_result,
        env_idx,
        agent_name,
    ):
        """构建某个 agent 专属的 reflection prompt。

        - solver: 用 memory_history_log（无 agent 标签的原始 CDMem 格式）
        - ground_truth: 用 history_log（带 [solver]/[ground_truth] 标签的版本），
          让纠正反思能区分 solver 自己的动作和 ground_truth 的介入。
        """
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
            memory_history_log,
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
        - per-agent local memory log（各自的 reflection 链）
        - trial log 使用带 agent 名称的 history（方便人类阅读）
        """
        for trial_idx in range(self.start_trial_num, self.num_trials):
            self.logger.log_world_start(trial_idx)
            num_successes = 0
            num_additional_successes = 0

            for env_idx in range(self.num_envs):
                init_ob, info = self.env.reset()
                print(f"{env_idx} using {self.env.name}")

                # 跳过 start_env_num 之前的环境
                if env_idx < self.start_env_num:
                    print(
                        f"  [skip] env_idx={env_idx} < start_env_num={self.start_env_num}")
                    self.logger.log_trial_content(
                        "[skipped]", False, trial_idx, env_idx)
                    continue

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
                    # 如果是solver则用无标签版本，ground truth则用带标签版本
                    history_log,
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
