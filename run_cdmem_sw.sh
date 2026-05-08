source ./env.sh
python src/main.py \
        --num_trials 4 \
        --num_envs 20 \
        --run_name "scworld_qwen3.6-flash_30_steps" \
        --model "qwen/qwen3.6-flash"  \
        --agent "cdmem"   \
        --env "scienceworld" \
        --start_trial_num 0 \
        --max_steps 20 \
        # --is_resume \
        # --resume_dir logs/scworld_qwen3.6-flash_20_steps_0506_133226 \
        
