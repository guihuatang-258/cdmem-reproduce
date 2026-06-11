from alfworld.agents.environment import get_environment
import alfworld
import yaml
import importlib
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
LOCAL_ALFWORLD_SRC = os.path.join(REPO_ROOT, "data", "alfworld")
if LOCAL_ALFWORLD_SRC not in sys.path:
    sys.path.insert(0, LOCAL_ALFWORLD_SRC)
os.environ.setdefault("ALFWORLD_DATA", os.path.join(REPO_ROOT, "data"))

# import alfworld.agents.environment


class AlfworldEnv:
    def __init__(self):
        importlib.reload(alfworld)
        # importlib.reload(alfworld.agents.environment)
        # 更改为最新目录
        with open('data/alfworld/configs/base_config.yaml') as reader:
            config = yaml.safe_load(reader)
        # split = "eval_out_of_distribution"
        # self.env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
        env_type = config['env']['type']  # 本实验使用AlfredTWEnv，即TextWorld纯文字实验
        self.env = get_environment(env_type)(
            config, train_eval="eval_out_of_distribution")
        self.env = self.env.init_env(batch_size=1)
        self.last_action = None

    def step(self, action):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(
            observation[0]), info['won'][0], done[0]
        # 如果当前动作是以'think:'开头的思考动作，则观察结果为'OK.'，奖励不变，游戏继续进行
        if action.startswith('think:'):
            observation = 'OK.'
        exhausted = False
        # ! 先注释掉exhausted，因为会和MAS冲突
        # if self.last_action == action:
        #     exhausted = True
        # else:
        #     self.last_action = action
        return observation, reward, done, exhausted, info

    def reset(self):
        self.last_action = None
        ob, info = self.env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        self.name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        return ob, info

    def reload(self):
        self.__init__()

    def close(self):
        self.env.close()

    def action_parser(self, action):
        action = action.strip()
        # *!强制只保留一行，避免模型幻觉
        first_line = action.split('\n')[0]
        if ">" in first_line:
            first_line = first_line.replace(">", "").strip()
        first_line = re.sub(r"^\[[^\]]+\]\s*", "", first_line).strip()
        first_line = re.sub(r"\s+", " ", first_line).strip()
        # 处理 put 动作，替换为 move 动作
        put_match = re.match(
            r"^put\s+(.+?)\s+(?:in/on|into|onto|in|on)\s+(.+)$",
            first_line,
            re.IGNORECASE,
        )
        if put_match:
            obj, receptacle = put_match.groups()
            first_line = f"move {obj.strip()} to {receptacle.strip()}"
        return first_line


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob
