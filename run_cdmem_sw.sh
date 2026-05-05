source ./env.sh
python src/main.py \
        --num_trials 5 \
        --num_envs 50 \
        --run_name "scworld_qwen3.6-flash_30_steps" \
        --model "qwen/qwen3.6-flash"  \
        --agent "cdmem"   \
        --env "scienceworld" \
        --start_trial_num 0
        # --max_steps 30
        # --is_resume \
        # --resume_dir logs/CDMem_logs_t5e67_0628_050752 \
        
