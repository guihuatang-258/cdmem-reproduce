python src/main.py \
        --num_trials 3 \
        --num_envs 1 \
        --run_name "expel_logs_t5e134_test" \
        --model "qwen/qwen3.5-flash-02-23"  \
        --agent "expel"   \
        --env "alfworld"
        # --is_resume \
        # --resume_dir logs/expel_logs_t5e67_0806_051212 \
        # --start_trial_num 1