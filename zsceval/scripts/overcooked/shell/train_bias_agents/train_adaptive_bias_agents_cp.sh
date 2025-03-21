#!/bin/bash
env="Overcooked"

layout=$1
weight_pattern=$2
population_size=$3
opponent=$4

entropy_coefs="0.2 0.05 0.001"
entropy_coef_horizons="0 18e6 3e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.001"
    entropy_coef_horizons="0 8e6 1e7"
fi
reward_shaping_horizon="3e7"
num_env_steps="3e7"

num_agents=2
algo="mappo_cp"
pop=hsp_${weight_pattern}_shared
exp="adaptive_${pop}-${opponent}_cp-s${population_size}"
path=../../policy_pool

export POLICY_POOL=${path}
n_training_threads=100

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
    # old layouts
    #! positive reward shaping for "[op]_X" may crash the training, be careful
    #! negative reward shaping for "put_X" may be meaningless
    # "put_onion_on_X",
    # "put_dish_on_X",
    # "put_soup_on_X",
    # "pickup_onion_from_X", random0_medium random0_hard
    # "pickup_onion_from_O", all_old
    # "pickup_dish_from_X",
    # "pickup_dish_from_D", all_old
    # "pickup_soup_from_X", random0 random0_medium random0_hard
    # "USEFUL_DISH_PICKUP", default
    # "SOUP_PICKUP", all_old default
    # "PLACEMENT_IN_POT", all_old default
    # "delivery", all_old
    # "STAY", all_old
    # "MOVEMENT",
    # "IDLE_MOVEMENT",
    # "IDLE_INTERACT_X",
    # "IDLE_INTERACT_EMPTY",
    # sparse_reward all_old
    
    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    if [[ "${layout}" == "random0" ]]; then
        #w0="0,0,[-5:0:5],0,0,0,0,0,[-5:0:5],0,0,3,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
        w0="0,0,5,0,0,0,0,0,-5,0,0,3,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
        seed_begin=1
        seed_max=1
    elif [[ "${layout}" == "random0_medium" ]]; then
        w0="0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=54
    elif [[ "${layout}" == "small_corridor" ]]; then
        w0="0,0,0,0,[-20:0:5],0,[-20:0:5],0,3,5,3,[-20:0],[-0.1:0],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=124
    else
        w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=176
    fi
else 
    version="new"
    # 0 "put_onion_on_X",　　0
    # 1 "put_tomato_on_X",　0
    # 2 "put_dish_on_X",　　[-5:0:5]
    # 3 "put_soup_on_X",　　0
    # 4 "pickup_onion_from_X",　　0
    # 5 "pickup_onion_from_O",　　0
    # 6 "pickup_tomato_from_X",　0
    # 7 "pickup_tomato_from_T",  0
    # 8 "pickup_dish_from_X",   [-5:0:5]
    # 9 "pickup_dish_from_D",  0
    # 10 "pickup_soup_from_X", 0
    # 11 "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter   3
    # 12 "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)    5
    # 13 "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot   3
    # 14 "viable_placement",   0
    # 15 "optimal_placement",  0
    # 16 "catastrophic_placement",  0
    # 17 "useless_placement",  0   # pot an ingredient to a useless recipe
    # 18 "potting_onion",  [-20:0]
    # 19 "potting_tomato",  [-20:0]
    # 20 "cook",   0
    # 21 "delivery",    0
    # 22 "deliver_size_two_order",   [-5:0:20]
    # 23 "deliver_size_three_order",   [-15:0:10]
    # 24 "deliver_useless_order",  0 
    # 25 "STAY",   [-0.1:0:0.1]
    # 26 "MOVEMENT",    0
    # 27 "IDLE_MOVEMENT", 0
    # 28 "IDLE_INTERACT",  0
    # 29 sparse_reward  1

    if [[ "${weight_pattern}" == "plate" ]]; then
        w0="0,0,[-5:0:5],0,0,0,0,0,[-5:0:5],0,0,3,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
        # w0="0,0,5,0,0,0,0,0,-5,0,0,3,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
        seed_begin=1
        seed_max=10
    elif [[ "${weight_pattern}" == "random0_medium" ]]; then
        w0="0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=54
    elif [[ "${weight_pattern}" == "all" ]]; then
        w0="[-3:0:3],[-3:0:3],[-3:0:3],[-3:0:3],[-3:0:3],[-3:0:3],[-3:0:3],[-3:0:3],[-3:0:3],0,0,3,5,3,0,0,0,0,[-20:0],[-20:0],0,0,[-5:0:20],[-15:0:10],0,[-0.1:0:0.1],0,0,0,[-1:0:1]"
        seed_begin=1
        seed_max=10000
    else
        w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=72
    fi

    we0=$w0
    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    we1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

fi

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    yml_path="${path}/${layout}/hsp/cp/${pop}/train-s${population_size}-${seed}.yml"
    echo "seed is ${seed}:"
    python train/train_adaptive_bias_agent.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --n_rollout_threads ${n_training_threads} --dummy_batch_size 1 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 5 \
    --population_yaml_path ${yml_path} \
    --population_size ${population_size} --adaptive_agent_name hsp_cp --use_agent_policy_id \
    --use_proper_time_limits \
    --use_hsp --use_expectation --w0 ${w0} --w1 ${w1} --we0 ${we0} --we0_offset 1 --we1 ${we1} --random_index \
    --wandb_name "hogebein"
done