source ./env.sh
python src/main.py \
        --num_trials 5 \
        --num_envs 50 \
        --run_name "scworld_logs_CDMem_add_env" \
        --model "google/gemini-3.1-pro-preview"  \
        --agent "cdmem"   \
        --env "scienceworld" \
        --start_trial_num 0
        # --is_resume \
        # --resume_dir logs/CDMem_logs_t5e67_0628_050752 \
        
