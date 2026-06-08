import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from .autogen.autogen_prompt import AUTOGEN_PROMPT
from .cdmem import CDMemAgent


@dataclass
class CDMemAutoGenMember:
    name: str
    role: str
    system_instruction: str
    local_memory: object
    global_memory: object
    logging_dir: str

    def response(self, llm, user_prompt: str, stop=None, max_tokens: int = 256) -> str:
        return llm(
            user_prompt,
            stop=stop,
            max_tokens=max_tokens,
            sys_msg=self.system_instruction,
        )


class CDMemAutoGenTeam:
    """A lightweight MetaMAS-style team that keeps CDMem's runtime stack intact."""

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
    CDMem for ALFWorld with an AutoGen/MAS-style multi-agent scheduler.

    The environment, prompt builder, few-shot builder, and global memory are kept
    identical to the original CDMem pipeline. The agent team follows the provided
    AutoGen structure: a solver acts by default, and a ground-truth helper is
    called only when the solver is stuck in a repeated-action loop. Short memory
    is shared as the team trajectory; local and long memories are per agent.
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
        self.local_memory_cls = local_memory
        self.global_memory_cls = global_memory
        self.agent_memory_root = os.path.join(self.logging_dir, "agent_memory")
        os.makedirs(self.agent_memory_root, exist_ok=True)
        self.team = self._build_team()

    def _build_team(self) -> CDMemAutoGenTeam:
        action_contract = self._action_system_message()
        role_specs = [
            (
                "solver",
                "solver",
                AUTOGEN_PROMPT.solver_system_prompt,
            ),
            (
                "ground_truth",
                "ground truth agent",
                AUTOGEN_PROMPT.ground_truth_system_prompt,
            ),
        ]
        team = CDMemAutoGenTeam()
        members = []
        for name, role, instruction in role_specs:
            agent_logging_dir = os.path.join(self.agent_memory_root, name)
            os.makedirs(agent_logging_dir, exist_ok=True)
            members.append(
                CDMemAutoGenMember(
                    name=name,
                    role=role,
                    system_instruction=f"{instruction.strip()}\n\n{action_contract}",
                    local_memory=self.local_memory_cls(self.num_envs),
                    global_memory=self.global_memory_cls(
                        agent_logging_dir,
                        getattr(self, "is_vector", False),
                    ),
                    logging_dir=agent_logging_dir,
                )
            )
        team.hire(
            members
        )
        return team

    def run_trajectory(self, env_idx, init_ob, to_print=True):
        cur_step = 0
        turn_idx = 0
        action_history: List[str] = []
        active_agent_name = "solver"

        print(init_ob)
        self.short_memory.reset()

        max_turns = max(self.max_steps * 2, self.max_steps)
        while cur_step < self.max_steps and turn_idx < max_turns:
            solver = self.team.get_agent("solver")
            active_agent = solver
            active_agent_name = "solver"
            infer_prompt = self.build_infer_prompt(env_idx, init_ob, active_agent_name)

            action = active_agent.response(
                self.llm,
                infer_prompt,
                stop=["\n"],
            ).strip()
            action = self.env.action_parser(action)
            action = self.normalize_place_action(action)
            if not action:
                action = "think: I need to choose a valid next action."

            if self._solver_stuck(action, action_history):
                active_agent_name = "ground_truth"
                active_agent = self.team.get_agent(active_agent_name)
                infer_prompt = self.build_infer_prompt(env_idx, init_ob, active_agent_name)
                action = active_agent.response(
                    self.llm,
                    infer_prompt,
                    stop=["\n"],
                ).strip()
                action = self.env.action_parser(action)
                action = self.normalize_place_action(action)
                if not action:
                    action = "think: I need to choose a valid next action."

            self._add_short_memory("action", action, active_agent_name)
            observation, reward, done, exhausted, info = self.env.step(action)
            observation_text = "Observation: " + observation
            self._add_short_memory("observation", observation_text, active_agent_name)

            if to_print:
                print(f"[{active_agent_name}] Action: {action}")
                print(f"Observation: {observation}")
                sys.stdout.flush()

            action_history.append(action)
            turn_idx += 1

            if done:
                history_log = self.build_infer_prompt(env_idx, init_ob, active_agent_name)
                return history_log, True
            elif exhausted:
                history_log = self.build_infer_prompt(env_idx, init_ob, active_agent_name)
                return history_log, False
            if action.startswith("think:"):
                continue
            cur_step += 1

        history_log = self.build_infer_prompt(env_idx, init_ob, active_agent_name)
        return history_log, False

    def _solver_stuck(self, current_action: str, action_history: List[str]) -> bool:
        return (
            len(action_history) >= 2
            and current_action == action_history[-1]
            and current_action == action_history[-2]
        )

    def _add_short_memory(self, label: str, value: str, active_agent_name: str) -> None:
        self.short_memory.add(label, value)

    def build_infer_prompt(self, env_idx, init_ob, agent_name: Optional[str] = None):
        short_memories = self.short_memory.recall()
        local_memories = self._recall_agent_local_memories(agent_name, env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]

        env_description, task_description = self.process_before_infer(init_ob)
        agent_global_memory = self._get_agent_global_memory(agent_name)
        agent_logging_dir = self._get_agent_logging_dir(agent_name)
        known_obs_history, action_guidance_history = agent_global_memory.recall(
            env_description, task_description
        )
        fewshots = self.fewshot_builder.get_inference_fewshots(
            self.env.name,
            env_description,
            task_description,
            agent_global_memory,
            agent_logging_dir,
        )
        return self.prompt_builder.get_inference_prompts(
            init_ob,
            fewshots,
            local_memories,
            short_memories,
            known_obs_history,
            action_guidance_history,
        )

    def _recall_agent_local_memories(self, agent_name: Optional[str], env_idx: int):
        if agent_name:
            agent = self.team.get_agent(agent_name)
            if agent is not None:
                return agent.local_memory.recall(env_idx)
        return self.local_memory.recall(env_idx)

    def _get_agent_global_memory(self, agent_name: Optional[str]):
        if agent_name:
            agent = self.team.get_agent(agent_name)
            if agent is not None:
                return agent.global_memory
        return self.global_memory

    def _get_agent_logging_dir(self, agent_name: Optional[str]):
        if agent_name:
            agent = self.team.get_agent(agent_name)
            if agent is not None:
                return agent.logging_dir
        return self.logging_dir

    def update_local_memory(self, history_log, is_success, env_idx):
        if self.local_memory.is_skip(env_idx):
            return self._empty_expert_trajectory(history_log, is_success)

        expert_prompt = self.build_expert_prompt(history_log)
        expert_result = self.llm(expert_prompt, max_tokens=512)
        print(f"[DEBUG] expert_result: {expert_result}")

        agent_trajectories = {}
        combined_reflections = []
        for agent_name, agent in self.team.agents_team.items():
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
            print(f"[DEBUG] {agent_name} reflection_result: {reflection_result}")
            expert_trajectory = self.process_after_reflection(
                expert_result,
                reflection_result,
                history_log,
                is_success,
            )
            agent.local_memory.add(env_idx, expert_trajectory)
            agent_trajectories[agent_name] = expert_trajectory
            combined_reflections.append(
                f"{agent_name} ({agent.role}): {reflection_result.strip()}"
            )

        canonical_reflection = "\n".join(combined_reflections)
        canonical_trajectory = self.process_after_reflection(
            expert_result,
            canonical_reflection,
            history_log,
            is_success,
        )
        canonical_trajectory["agent_reflections"] = agent_trajectories
        self.local_memory.add(env_idx, canonical_trajectory)
        print(f"[DEBUG] canonical local memory={self.local_memory.recall(env_idx)}")
        return canonical_trajectory

    def build_reflection_prompt_for_agent(
        self,
        history_log,
        is_success,
        expert_result,
        env_idx,
        agent_name,
    ):
        local_memories = self._recall_agent_local_memories(agent_name, env_idx)
        if len(local_memories) > 3:
            local_memories = local_memories[-3:]
        fewshots = self.fewshot_builder.get_reflection_fewshots(is_success)
        return self.prompt_builder.get_reflection_prompts(
            history_log,
            is_success,
            fewshots,
            local_memories,
            expert_result,
        )

    def _empty_expert_trajectory(self, history_log, is_success):
        env_description = task_description = ""
        scenario = history_log.split("Here is the task:")[-1].strip()
        for line in scenario.splitlines():
            if line.startswith("You are in the middle of a room."):
                env_description = line.strip()
            elif line.startswith("Your task is to:"):
                task_description = line.split("Your task is to:", 1)[-1].strip()
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
        import json

        for agent_name, agent in self.team.agents_team.items():
            path = os.path.join(
                agent.logging_dir,
                f"local_memory_trial_{trial_idx}.json",
            )
            with open(path, "w") as wf:
                json.dump(agent.local_memory.history, wf, indent=4)

    def log_agent_global_memory(self, trial_idx: int) -> None:
        import json

        for agent_name, agent in self.team.agents_team.items():
            env_path = os.path.join(
                agent.logging_dir,
                f"global_env_trial_{trial_idx}.json",
            )
            task_path = os.path.join(
                agent.logging_dir,
                f"global_task_trial_{trial_idx}.json",
            )
            with open(env_path, "w") as wf:
                json.dump(agent.global_memory.env_memory, wf, indent=4)
            with open(task_path, "w") as wf:
                json.dump(agent.global_memory.task_memory, wf, indent=4)

    def update_agent_global_memories(self, canonical_trajectory, env_idx, trial_idx):
        agent_trajectories = canonical_trajectory.get("agent_reflections", {})
        for agent_name, expert_trajectory in agent_trajectories.items():
            agent = self.team.get_agent(agent_name)
            if agent is None:
                continue
            self.update_global_memory_for_agent(
                agent,
                expert_trajectory,
                env_idx,
                trial_idx,
            )

    def update_global_memory_for_agent(self, agent, expert_trajectory, env_idx, trial_idx):
        env_query = task_query = ""
        increment_env, increment_task = agent.global_memory.short2long(
            expert_trajectory,
            env_idx,
            trial_idx,
        )
        is_success = expert_trajectory["is_success"]
        if len(increment_env) != 0:
            env_fewshots = self.fewshot_builder.get_summary_fewshots("env")
            env_query = self.prompt_builder.env_summary_prompts(
                increment_env,
                env_fewshots,
            )
        if len(increment_task) != 0:
            task_fewshots = self.fewshot_builder.get_summary_fewshots(
                "task",
                is_success,
            )
            task_query = self.prompt_builder.task_summary_prompts(
                increment_task,
                task_fewshots,
                is_success,
            )
        if env_query:
            env_summary = self.llm(env_query, max_tokens=512)
            agent.global_memory.add(env_summary, expert_trajectory, mode="env")
        if task_query:
            task_summary = self.llm(task_query, max_tokens=512)
            agent.global_memory.add(task_summary, expert_trajectory, mode="task")

    def run(self):
        for trial_idx in range(self.start_trial_num, self.num_trials):
            self.logger.log_world_start(trial_idx)
            num_successes = 0
            num_additional_successes = 0

            for env_idx in range(self.num_envs):
                init_ob, info = self.env.reset()
                print(f"{env_idx} using {self.env.name}")
                if self.local_memory.is_success(env_idx):
                    num_successes += 1
                    self.logger.log_world_success(trial_idx, env_idx)
                    self.logger.log_trial_success(trial_idx, env_idx)
                    continue

                history_log, is_success = self.run_trajectory(env_idx, init_ob)
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
                    history_log,
                    is_success,
                    trial_idx,
                    env_idx,
                )
                expert_trajectory = self.update_local_memory(
                    history_log,
                    is_success,
                    env_idx,
                )
                self.logger.log_local_memory(trial_idx)
                self.log_agent_local_memory(trial_idx)
                self.update_global_memory(expert_trajectory, env_idx, trial_idx)
                self.update_agent_global_memories(expert_trajectory, env_idx, trial_idx)
                self.logger.log_global_memory(trial_idx)
                self.log_agent_global_memory(trial_idx)

            self.env.close()
            self.logger.log_world_end(
                trial_idx,
                num_successes,
                num_additional_successes,
            )
            self.env.reload()

    def _action_system_message(self) -> str:
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
- examine (something): examine a receptacle or an object
- use (object): use an object
- heat (object) with (receptacle): heat an object using a receptacle
- clean (object) with (receptacle): clean an object using a receptacle
- cool (object) with (receptacle): cool an object using a receptacle
- slice (object) with (object): slice an object using a sharp object

Note: Use exact object and receptacle names with their numbers from observations, such as "apple 1", "drawer 2", or "sinkbasin 1". Do not invent object names, receptacle names, ids, or environment feedback.
Important: Do not output "put ..." commands. In this ALFWorld TextWorld environment, placing an object must use "move (object) to (receptacle)", for example "move apple 1 to drawer 2".

Please note, the task interactive trajectory is realtime feedback from environment. You are required to interact with the environment to complete the task.
So, you need to output your thinking or a valid action, and the action will be executed in the environment.
- If you output your thinking, the environment will simple respond with "OK".
- If you output a valid action, the environment will return a new observation.
- If you meet operation failure, the environment will return "Nothing happens", which means the current observation doesn't match the current action. Please rethink and choose a different action.
""".strip()
