python src/main.py \
        --num_trials 1 \
        --num_envs 1 \
        --run_name "cdmem" \
        --model "qwen/qwen3.5-plus-02-15"  \
        --agent "cdmem"   \
        --env "alfworld"
        # --model "qwen/qwen3.5-flash-02-23"  \
        # --model "qwen/qwen3.5-plus-02-15"  \

# python src/main.py \
#         --num_trials 1 \
#         --num_envs 1 \
#         --run_name "scworld_logs_CDMem_add_env" \
#         --model "qwen/qwen3.5-flash-02-23"  \
#         --agent "cdmem"   \
#         --env "scienceworld" \
#         --start_trial_num 0