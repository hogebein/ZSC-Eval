#!/bin/bash
env="Overcooked"

layout=$1

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

num_agents=2
algo="population"

if [[ $2 == "fcp" ]];
then
    algorithm="fcp"
    # exps=("fcp-S2-s12")
    exps=("fcp-S2-s24" "fcp-S2-s36")
    # exps=("fcp-S2-s12" "fcp-S2-s24" "fcp-S2-s36")
    exps=("fcp-S2-s24" "fcp-S2-s36")
elif [[ $2 == "mep" ]];
then
    algorithm="mep"
    # exps=("mep-S2-s12")
    exps=("mep-S2-s24" "mep-S2-s36")
    # exps=("mep-S2-s12" "mep-S2-s24" "mep-S2-s36")
    exps=("mep-S2-s24" "mep-S2-s36")
elif [[ $2 == "traj" ]];
then
    algorithm="traj"
    exps=("traj-S2-s24" "traj-S2-s36")
    # exps=("traj-S2-s12" "traj-S2-s24" "traj-S2-s36")
    exps=("traj-S2-s24" "traj-S2-s36")
elif [[ $2 == "hsp" ]];
then
    algorithm="hsp"
    # exps=("hsp-S2-s24" "hsp-S2-s36")
    #exps=("hsp-S2-s12" "hsp-S2-s24" "hsp-S2-s36")

    exps=("hsp_onion_tomato_shared-S2-s10")
    pop_agent_version="hsp_onion_tomato_shared"
elif [[ $2 == "cole" ]];
then
    algorithm="cole"
    exps=("cole-S2-s50" "cole-S2-s75")
    # exps=("cole-S2-s25" "cole-S2-s50" "cole-S2-s75")
    exps=("cole-S2-s50" "cole-S2-s75")
elif [[ $2 == "hsp_cp" ]];
then
    algorithm="hsp_cp"
    #exps=("adaptive_hsp_plate_shared-pop_cp-s60" "hsp_plate_shared-pop_cp-s60")
    #exps=("mep-S2-s36-adp_cp-s5" "hsp_plate-S2-s36-adp_cp-s5" "adaptive_mep-S2-s36-adp_cp-s5" "adaptive_hsp_plate-S2-s36-adp_cp-s5")

elif [[ $2 == "hsp_react" ]];
then
    algorithm="hsp"
    #exps=("hsp_plate_placement_shared-S2-s10")
    #exps=("reactive_hsp_plate_placement_shared-S3-s10")
    #pop_agent_version="hsp_plate_placement_shared"
    #exps=("hsp_plate_placement_shared-S2-s5")
    #pop_agent_version="hsp_plate_placement_shared"

else
    echo "bash eval_with_bias_agents.sh {layout} {algo}"
    exit 0
fi


pop_agent_algo="hsp"

declare -A LAYOUTS_KS
LAYOUTS_KS["random0"]=10
LAYOUTS_KS["random0_medium"]=10
LAYOUTS_KS["random1"]=10
LAYOUTS_KS["random3"]=10
LAYOUTS_KS["small_corridor"]=10
LAYOUTS_KS["unident_s"]=10
LAYOUTS_KS["random0_m"]=10
LAYOUTS_KS["random1_m"]=10
LAYOUTS_KS["random3_m"]=10
LAYOUTS_KS["random3_mm"]=10
LAYOUTS_KS["placement_coordination"]=10

path=../../policy_pool
export POLICY_POOL=${path}

#K=$((LAYOUTS_KS[${layout}]))
K=10
bias_yml="${path}/${layout}/${pop_agent_algo}/s1/${pop_agent_version}/benchmarks-s${K}.yml"
yml_dir=eval/eval_policy_pool/${layout}/results
mkdir -p ${yml_dir}

n=$(grep -o -E 'bias.*_(final|mid):' ${bias_yml} | wc -l)
echo "Evaluate ${layout}, ${exps} with ${n} agents"
population_size=$((n + 1))

ulimit -n 65536

eval_exp_v=$3

n_seed=1
rollout_threads=1
fixed_index=1

options=()

w0_path="${path}/${layout}/${algorithm}/s1/${pop_agent_version}/w0.json"
for (( i=$K+1; i>0; i-- )); do
    w0_i=$(cat ${w0_path} | jq ".${algorithm}${i}_final_actor")
    options+=(-e "s/@@${i}/${w0_i}/g")
done

len=${#exps[@]}
for (( i=0; i<$len; i++ )); do
    exp=${exps[$i]}

    echo "Evaluate population ${algo} ${exp} ${population}"
    for seed in $(seq 1 $((n_seed))); do
        exp_name="${exp}"
        agent_name="${exp_name}-${seed}"
        
        echo "Exp name ${exp_name} ${eval_exp_v}"

        if [[ "${eval_exp_v}" == "reactive" ]];
        then

            echo "use_reactive"

            eval_exp="eval_cp-${eval_exp_v}-${agent_name}"
            yml=${yml_dir}/${eval_exp}.yml

            sed -e "s/agent_name/${agent_name}/g" -e "s/algorithm/${algorithm}/g" -e "s/population/${exp_name}/g" -e "s/seed/${seed}/g" "${options[@]}" "${bias_yml}" > "${yml}"

            python eval/eval_with_population.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${eval_exp}" --layout_name "${layout}" \
            --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads "${rollout_threads}" --eval_episodes 20 --eval_stochastic --dummy_batch_size 1 \
            --use_proper_time_limits \
            --use_wandb \
            --store_traj \
            --population_yaml_path "${yml}" --population_size ${population_size} \
            --overcooked_version ${version} --eval_result_path "eval/results/${layout}/${algorithm}/${eval_exp}.json" \
            --agent_name "${agent_name}" \
            --use_reactive \
            --use_agent_policy_id \
            --use_opponent_utility \
            --use_render \
            --reaction_type 1 \
            --filter_type 0 \
            --fixed_index ${fixed_index}
        else

            eval_exp="eval_cp-${agent_name}"
            yml=${yml_dir}/${eval_exp}.yml

            sed -e "s/agent_name/${agent_name}/g" -e "s/algorithm/${algorithm}/g" -e "s/population/${exp_name}/g" -e "s/seed/${seed}/g" "${options[@]}" "${bias_yml}" > "${yml}"

            python eval/eval_with_population.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${eval_exp}" --layout_name "${layout}" \
            --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads "${rollout_threads}" --eval_episodes 20 --eval_stochastic --dummy_batch_size 1 \
            --use_proper_time_limits \
            --use_wandb \
            --store_traj \
            --population_yaml_path "${yml}" --population_size ${population_size} \
            --overcooked_version ${version} --eval_result_path "eval/results/${layout}/${algorithm}/${eval_exp}.json" \
            --agent_name "${agent_name}" \
            --use_agent_policy_id \
            --use_opponent_utility \
            --use_render \
            --fixed_index ${fixed_index}

        fi

        
    done
done
