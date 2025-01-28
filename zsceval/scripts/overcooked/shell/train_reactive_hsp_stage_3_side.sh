
#!/bin/bash
env="Overcooked"

layout=$1
population_size=$2
index=$3
pop_version=$4

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

use_base_shaping_r=true

if [[ ${population_size} == 5 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 2.5e7 5e7"
    if [[ ${pop_version} == "tomato_delivery" ]]; then
        pop="hsp_tomato_delivery_shared"
        reward_shaping_horizon="15e7"
        num_env_steps="15e7"
        use_base_shaping_r=true
    elif [[ ${pop_version} == "tomato_lover" ]]; then
        pop="hsp_onion_tomato_shared"
        reward_shaping_horizon="5e7"
        num_env_steps="5e7"
        use_base_shaping_r=true
    elif [[ ${pop_version} == "score" ]]; then
        reward_shaping_horizon="5e7"
        num_env_steps="5e7"
        pop="hsp_score"
        use_base_shaping_r=true
    else
        pop="hsp_plate_placement_shared"
        reward_shaping_horizon="5e7"
        num_env_steps="5e7"
        use_base_shaping_r=true
    fi
    mep_exp="no_mep"

elif [[ ${population_size} == 10 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 2.5e7 5e7"

    if [[ ${pop_version} == "tomato_delivery" ]]; then
        pop="hsp_tomato_delivery_shared"
        reward_shaping_horizon="15e7"
        num_env_steps="15e7"
        use_base_shaping_r=true

        filter_type=2
    elif [[ ${pop_version} == "onion_tomato" ]]; then
        pop="hsp_onion_tomato_shared"
        reward_shaping_horizon="10e7"
        num_env_steps="10e7"
        use_base_shaping_r=true

        filter_type=1
    else
        pop="hsp_plate_placement_shared"
        reward_shaping_horizon="5e7"
        num_env_steps="5e7"
        use_base_shaping_r=true
        
        filter_type=0
    fi
    mep_exp="no_mep"
    
elif [[ ${population_size} == 24 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 4e7 8e7"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 6.4e7 8e7"
    fi
    reward_shaping_horizon="8e7"
    num_env_steps="8e7"
    pop="hsp"
    mep_exp="mep-S1-s10"
elif [[ ${population_size} == 36 ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 5e7 1e8"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 8e7 1e8"
    fi
    reward_shaping_horizon="1e8"
    num_env_steps="1e8"
    pop="hsp"
    mep_exp="mep-S1-s15"
else
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 5e7 1e8"
    if [[ "${layout}" == "small_corridor" ]]; then
        entropy_coefs="0.2 0.05 0.01"
        entropy_coef_horizons="0 8e7 1e8"
    fi
    reward_shaping_horizon="1e8"
    num_env_steps="1e8"
    pop="hsp"
    mep_exp="mep-S1-s10"
fi


num_agents=2
algo="adaptive"
exp="reactive_${pop}-S3-s${population_size}"
stage="S2"
seed_begin=1
seed_max=1
path=../../policy_pool

export POLICY_POOL=${path}

n_training_threads=200

ulimit -n 65536

reaction_type=0

echo "env is ${env}, layout is ${layout}, algo is ${algo}, pop is ${pop}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}, stage is ${stage}"
for seed in $(seq ${seed_begin} ${seed_max});
# for seed in 1 2 5;
do
    if "${use_base_shaping_r}"; then

    	python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    	--seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    	--overcooked_version ${version} \
    	--n_rollout_threads ${n_training_threads} --dummy_batch_size 1 \
    	--ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    	--stage 2 \
    	--save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 20 \
    	--population_yaml_path ${path}/${layout}/hsp_react/s3/train-s${population_size}-${pop}_${mep_exp}-${seed}.yml \
    	--population_size ${population_size} --adaptive_agent_name hsp_adaptive --use_agent_policy_id \
    	--use_proper_time_limits \
    	--wandb_name "hogebein" \
    	--use_reactive \
    	--use_opponent_utility \
    	--use_base_shaping_r \
        --fixed_index ${index} \
        --cuda_id 1 \
        --reaction_type ${reaction_type} \
        --filter_type ${filter_type}
	    
    else

    	python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    	--seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    	--overcooked_version ${version} \
    	--n_rollout_threads ${n_training_threads} --dummy_batch_size 1 \
    	--ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    	--stage 2 \
    	--save_interval 25 --log_interval 1 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 20 \
    	--population_yaml_path ${path}/${layout}/hsp_react/s3/train-s${population_size}-${pop}_${mep_exp}-${seed}.yml \
    	--population_size ${population_size} --adaptive_agent_name hsp_adaptive --use_agent_policy_id \
    	--use_proper_time_limits \
    	--wandb_name "hogebein" \
    	--use_reactive \
        --use_opponent_utility \
        --fixed_index ${index} \
	    --cuda_id 1 \
        --reaction_type ${reaction_type} \
        --filter_type ${filter_type}

    fi
done

