source ./set_api.sh
cd ..
python src/main.py \
        --num_trials 3 \
        --num_envs 3 \
        --run_name "cdmem" \
        --model "qwen/qwen3-235b-a22b"  \
        --agent "cdmem"   \
        --env "alfworld" \
        # --is_vector
        # --is_resume \
        # --resume_dir logs/ \
        # --start_trial_num 4
        