python src/main.py \
        --num_trials 3 \
        --num_envs 10 \
        --run_name "scworld_qwen3.5-flash-02-23_20_steps" \
        --model "qwen/qwen3.5-flash-02-23"  \
        --agent "cdmem"   \
        --env "scienceworld" \
        --start_trial_num 0 \
        --max_steps 20 \
        # --is_resume \
        # --resume_dir logs/scworld_qwen3.6-flash_20_steps_0506_133226 \