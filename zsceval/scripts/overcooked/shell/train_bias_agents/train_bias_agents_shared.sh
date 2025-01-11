#!/bin/bash
env="Overcooked"

layout=$1
weight_pattern=$2

entropy_coefs="0.2 0.05 0.001"
entropy_coef_horizons="0 6e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.001"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e7"
num_env_steps="1e7"

num_agents=2
algo="mappo"
stage="S1"


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
# 29 "place_onion_on_X",
# 30 "place_tomato_on_X",
# 31 "place_dish_on_X",
# 32 "place_soup_on_X",
# 33 "recieve_onion_via_X",
# 34 "recieve_tomato_via_X",
# 35 "recieve_dish_via_X",
# 36 "recieve_soup_via_X",
# 37 "onions_placed_on_X",
# 38 "tomatoes_placed_on_X",
# 39 "dishes_placed_on_X",
# 40 "soups_placed_on_X",
# 41 "integral_onion_placed_on_X",
# 42 "integral_tomato_placed_on_X",
# 43 "integral_dish_placed_on_X",
# 44 "integral_soup_placed_on_X",
# 45 "deliver_onion_order",
# 46 "deliver_tomato_order",
# 47 "onion_order_delivered",
# 48 "tomato_order_delivered",
# 49 "sparse_reward  1,

w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
use_base_shaping_r=false

if [[ "${weight_pattern}" == "all" ]]; then
    w0="0,0,[-5:0:5],0,0,0,0,0,[-5:0:5],0,0,3,5,3,0,0,0,0,[-0.2:0],[-20:0],0,0,[-5:0:20],[-15:0:10],0,[-0.1:0:0.1],0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    seed_begin=1
    seed_max=124
elif [[ "${weight_pattern}" == "plate_placed" ]]; then

    if [[ "${layout}" == "random3_m" ]]; then
        w0="0,0,0,0,0,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,0,0,0,0,0,0,0,0,0,0"
    else
        w0="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,0,0,0,0,0,0,0,0,0,0"
    fi
    
    seed_begin=1
    seed_max=5

    reward_shaping_horizon="1e7"
    num_env_steps="1e7"

    exp="hsp_plate_placement_shared-${stage}"
elif [[ "${weight_pattern}" == "plate_placed_i" ]]; then
    if [[ "${layout}" == "random3_m" ]]; then
        w0="0,0,0,0,0,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,0,0,0,0,0,0,0,0,0,0"
    else
        w0="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,0,0,0,0,0,0,0,0,0,0"
    fi
    seed_begin=3
    seed_max=5

    reward_shaping_horizon="1e7"
    num_env_steps="1e7"

    exp="hsp_plate_placement_shared-${stage}"

elif [[ "${weight_pattern}" == "plate_place" ]]; then
    if [[ "${layout}" == "random3_m" ]]; then
        w0="0,0,0,0,0,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,0,0,0,0,0,0,0,0,0,0"
    else
        w0="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,0,0,0,0,0,0,0,0,0,0"
    fi
    seed_begin=6
    seed_max=10

    exp="hsp_plate_placement_shared-${stage}"

elif [[ "${weight_pattern}" == "tomato_state" ]]; then
   w0="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0"
   seed_begin=1
   seed_max=5
   exp="hsp_tomato_delivery_shared-${stage}"
   use_base_shaping_r=true

elif [[ "${weight_pattern}" == "tomato_self" ]]; then
   w0="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0"
   seed_begin=6
   seed_max=10
   exp="hsp_tomato_delivery_shared-${stage}"
   use_base_shaping_r=true

elif [[ "${weight_pattern}" == "score" ]]; then
    w0="0,0,0,0,0,0,0,0,0,3,0,0,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    seed_begin=5
    seed_max=5
    use_base_shaping_r=true
    exp="hsp_score-${stage}"

else
    #w0="0,0,0,0,0,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[0:3],0,0,0,0,0,0,0,[0:10],0,0,0,0,0,0"

    w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    seed_begin=1
    seed_max=72
fi

rollout_threads=80


echo "seed_max is ${seed_max}:"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"

    if "${use_base_shaping_r}"; then

        python train/train_bias_agent.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
        --seed ${seed} --n_training_threads 2 --n_rollout_threads ${rollout_threads} --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
        --overcooked_version ${version} \
        --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
        --use_hsp --w0 ${w0} --w1 ${w1} --random_index \
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
        --use_proper_time_limits \
        --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 20 \
        --wandb_name "hogebein" \
        --cuda_id 0 \
        --use_base_shaping_r
    
    
    else

        python train/train_bias_agent.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
        --seed ${seed} --n_training_threads 2 --n_rollout_threads ${rollout_threads} --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
        --overcooked_version ${version} \
        --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
        --use_hsp --w0 ${w0} --w1 ${w1} --random_index \
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
        --use_proper_time_limits \
        --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 20 \
        --wandb_name "hogebein" \
        --cuda_id 0


    fi

    
done
