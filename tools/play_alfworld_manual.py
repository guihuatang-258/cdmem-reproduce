#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ALFWORLD_SRC = REPO_ROOT / "data" / "alfworld"
ALFWORLD_DATA = REPO_ROOT / "data"
CONFIG_PATH = LOCAL_ALFWORLD_SRC / "configs" / "base_config.yaml"


def prefer_local_alfworld():
    os.environ["ALFWORLD_DATA"] = str(ALFWORLD_DATA)
    sys.path.insert(0, str(LOCAL_ALFWORLD_SRC))


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def action_parser(action):
    action = action.strip()
    first_line = action.split("\n")[0]
    if ">" in first_line:
        first_line = first_line.replace(">", "").strip()
    action_words = first_line.split(" ")
    if "put" in action_words:
        for i, word in enumerate(action_words):
            if word.strip().lower() in {"in", "on"}:
                action_words[i] = "in/on"
                first_line = " ".join(action_words)
    return first_line


def compact_name(gamefile):
    path = Path(gamefile)
    return "/".join(path.parts[-3:-1])


def load_raw_env(split):
    prefer_local_alfworld()

    try:
        from alfworld.agents.environment import get_environment
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing Python dependency: {missing}\n"
            "Run this script with the same Python/conda environment you use for run_cdmem.sh."
        ) from exc

    with CONFIG_PATH.open() as reader:
        config = yaml.safe_load(reader)

    env_type = config["env"]["type"]
    raw_env = get_environment(env_type)(config, train_eval=split)
    return raw_env


def resolve_gamefile(raw_env, args):
    if args.gamefile:
        gamefile = Path(args.gamefile).expanduser()
        if not gamefile.is_absolute():
            gamefile = REPO_ROOT / gamefile
        if gamefile.is_dir():
            gamefile = gamefile / "game.tw-pddl"
        if not gamefile.exists():
            raise FileNotFoundError(f"Game file not found: {gamefile}")
        return str(gamefile)

    if args.idx < 0 or args.idx >= len(raw_env.game_files):
        raise IndexError(f"--idx must be in [0, {len(raw_env.game_files) - 1}]")
    return raw_env.game_files[args.idx]


def list_games(raw_env, limit):
    total = len(raw_env.game_files)
    show = total if limit is None else min(limit, total)
    for i, gamefile in enumerate(raw_env.game_files[:show]):
        print(f"{i:04d}  {compact_name(gamefile)}")
        print(f"      {gamefile}")
    if show < total:
        print(f"... showing {show} of {total} games")


def play(gamefile, show_admissible):
    from alfworld.agents.environment import get_environment

    with CONFIG_PATH.open() as reader:
        config = yaml.safe_load(reader)

    raw_env = get_environment(config["env"]["type"])(
        config, train_eval="eval_out_of_distribution")
    raw_env.game_files = [gamefile]
    env = raw_env.init_env(batch_size=1)

    try:
        obs, info = env.reset()
        print(f"Game: {compact_name(info['extra.gamefile'][0])}")
        print(f"File: {info['extra.gamefile'][0]}")
        print(process_ob("\n".join(obs[0].split("\n\n")[1:])))

        while True:
            if show_admissible:
                print("\nAdmissible commands:")
                for command in info["admissible_commands"][0]:
                    print(f"  {command}")

            action = input("\n> ").strip()
            if action.lower() in {"quit", "exit"}:
                break

            action = action_parser(action)
            obs, reward, done, info = env.step([action])
            print(process_ob(obs[0]))
            print(f"won={info['won'][0]}, done={done[0]}")

            if done[0]:
                print("Completed.")
                break
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually play one ALFWorld TextWorld game from this repo.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Game index after config filtering and seed shuffle.")
    parser.add_argument("--gamefile", type=str,
                        help="Path to a game.tw-pddl file or its parent directory.")
    parser.add_argument("--split", default="eval_out_of_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
                        help="ALFWorld split used when listing or selecting by --idx.")
    parser.add_argument("--list", action="store_true",
                        help="List available games instead of playing.")
    parser.add_argument("--limit", type=int, default=50,
                        help="How many games to show with --list. Use -1 for all.")
    parser.add_argument("--show-admissible", action="store_true",
                        help="Print admissible commands before each input.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_env = load_raw_env(args.split)

    if args.list:
        list_games(raw_env, None if args.limit < 0 else args.limit)
        return

    gamefile = resolve_gamefile(raw_env, args)
    print(f"Using local alfworld source: {LOCAL_ALFWORLD_SRC}")
    print(f"Using ALFWORLD_DATA: {ALFWORLD_DATA}")
    print(f"Using config: {CONFIG_PATH}")
    play(gamefile, args.show_admissible)


if __name__ == "__main__":
    main()
