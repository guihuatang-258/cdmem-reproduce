python src/main.py \
        --num_trials 1 \
        --num_envs 3 \
        --run_name "smoke_cdmem_autogen" \
        --model "qwen/qwen-2.5-7b-instruct"  \
        --agent "cdmem_autogen"   \
        --env "alfworld" \
        --max_steps 20 \
        --is_vector
        # --is_resume \
        # --resume_dir logs/ \
        # --start_trial_num 0

        # Full ALFWorld run example:
        # --num_trials 5
        # --num_envs 134

        # Model examples:
        # --model "qwen/qwen3-235b-a22b"  \
        # --model "qwen/qwen3.5-flash-02-23"  \
        # --model "deepseek/deepseek-v3.2"  \
