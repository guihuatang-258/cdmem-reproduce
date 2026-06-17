python src/main.py \
        --num_trials 2 \
        --num_envs 50  \
        --run_name "cdmem_autogen_qwen-2.5-7b-instruct" \
        --model "qwen/qwen-2.5-7b-instruct"  \
        --agent "cdmem_autogen"   \
        --env "alfworld" \
        --max_steps 20 \
        --is_vector

        # --start_env_num 10 \


        # Full ALFWorld run example:
        # --num_trials 5
        # --num_envs 134

        # Model examples:
        # --model "qwen/qwen-2.5-7b-instruct"
