source ./env.sh
python src/main.py \
        --num_trials 4 \
        --num_envs 10 \
        --run_name "cdmem_deepseek-v4-flash_20_steps" \
        --model "deepseek/deepseek-v4-flash"  \
        --agent "cdmem"   \
        --env "alfworld" \
        --max_steps 20
        # --is_resume \
        # --resume_dir logs/cdmem_0503_100540 \
        # --start_trial_num 0

        # --model "qwen/qwen3.6-plus"  \
        # --model "qwen/qwen3.6-flash"  \
        # --model "qwen/qwen3.5-flash-02-23"  \
        # --model "qwen/qwen3.5-plus-20260420"  \
        # --model "deepseek/deepseek-v3.2"  \
        # --model "deepseek/deepseek-v4-flash"  \
        # --model "deepseek/deepseek-v4-pro"  \

        # --model "minimax/minimax-m2.7"  \
        # --model "moonshotai/kimi-k2.6"  \

# --num_trials 5
# --num_envs 134