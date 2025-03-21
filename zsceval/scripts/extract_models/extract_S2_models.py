import argparse
import os
import socket

import numpy as np
import wandb

wandb_name = "hogebein"
POLICY_POOL_PATH = "../policy_pool"

from loguru import logger


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_target_index(array, percentile: float):
    """
    Find the index of the target value in the array based on the percentile.
    array: numpy array
    percentile: float, between 0 and 1
    return: index of the target value in the array
    """
    q3 = np.nanpercentile(array, int(percentile * 100))

    array_without_nan = array[~np.isnan(array)]

    index_of_max = np.nanargmax(array_without_nan)
    logger.debug(f"max index {index_of_max}/{len(array_without_nan)}")

    filtered_array = array_without_nan[index_of_max + 1 :]

    if filtered_array.size > 0:
        relative_index = (np.abs(filtered_array - q3)).argmin()
        original_index = np.where(array == filtered_array[relative_index])[0][0]

        return original_index, q3
    else:
        return len(array) - 1, np.nanmax(array)



def extract_S2_models(layout, algorithm, exp, env, population: str):
    population = population.split("-", 1)[1]
    if algorithm == "fcp":
        exp_name = f"{exp}-pop{population}-S2"
    elif algorithm == "mep":
        if len(exp.split("-")) == 2:
            exp_name = f"{exp.split('-')[0]}-{exp.split('-')[1]}-pop{population}-S2"
        else:
            exp_name = f"{exp}_mep-S2-{population}_seed1"

    elif algorithm == "traj":
        exp_name = f"{exp}-pop{population}-S2"
    elif algorithm == "hsp":
        exp_name = f"{exp}-pop{population}-S2"
    elif algorithm == "cole":
        exp_name = f"{exp}-{population}"
    else:
        raise NotImplementedError

    logger.info(f"exp {exp_name}")
    api = wandb.Api()
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"
    runs = api.runs(
        f"{wandb_name}/{env}",
        # f"{wandb_name}/Overcooked-new",
        filters={
            "$and": [
                {"config.experiment_name": exp_name},
                #{"config.scenario_name": layout},
                #{"config.layout_name": layout},
                {"state": "finished"},
                {"tags": {"$nin": ["hidden", "unused"]}},
            ]
        },
        order="+config.seed",
    )
    #if not exp_name.endswith("-S2"):
    #    exp_name += "-S2"
    #    exp_name = exp + "-pop_" + exp_name.split(exp + "-", 1)[1]

def extract_pop_S2_models(layout, algo, exp, env, percentile=0.8, final=False):
    logger.info(f"exp {exp}")
    api = wandb.Api(timeout=60)
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"
    filters = {
        "$and": [
            {"config.experiment_name": exp},
            {layout_config: layout},
            {"state": "finished"},
            {"tags": {"$nin": ["hidden", "unused"]}},
        ]
    }
    logger.info(f"{wandb_name}/{env}")
    logger.info(f"filters {filters}")
    runs = api.runs(
        f"{wandb_name}/{env}",
        filters=filters,
        order="+config.seed",
    )

    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"num of runs: {len(runs)}")
    for i, run_id in enumerate(run_ids):
        run = runs[i]
        seed = run.config["seed"]
        if run.state == "finished":
            logger.info(f"Run: {run_id} Seed: {seed}")
            files = run.files()
            policy_name = f"{algo}_adaptive"
            history = run.history()
            history = history[["_step", f"either-{algo}_adaptive-ep_shaped_r"]]
            steps = history["_step"].to_numpy().astype(int)
            ep_shaped_r = history[f"either-{algo}_adaptive-ep_shaped_r"].to_numpy()
            
            files = run.files()
            actor_pts = [f for f in files if f.name.startswith(f"{policy_name}/actor_periodic")]
            actor_versions = [int(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts]
            actor_versions.sort()

            i_max_ep_shaped_r, max_ep_shaped_r = find_target_index(ep_shaped_r, percentile)
            max_ep_shaped_r_step = steps[i_max_ep_shaped_r]

            if final:
                version = find_nearest(actor_versions, actor_versions[-1] * percentile)
                logger.info(
                    f"final_version"
                )
                logger.info(
                    f"actor version {version} / {actor_versions[-1]}, shaped_r {max_ep_shaped_r:.3f}/{np.nanmax(ep_shaped_r):.3f}"
                )
            else:
                version = find_nearest(actor_versions, max_ep_shaped_r_step)
                logger.info(
                    f"actor version {version} / {actor_versions[-1]}, shaped_r {max_ep_shaped_r:.3f}/{np.nanmax(ep_shaped_r):.3f}"
                )
            ckpt = run.file(f"{policy_name}/actor_periodic_{version}.pt")
            tmp_dir = f"tmp/{layout}/{exp}"
            logger.info(f"Fetch {tmp_dir}/{policy_name}/actor_periodic_{version}.pt")
            ckpt.download(f"{tmp_dir}", replace=True)
            algo_s2_dir = f"{POLICY_POOL_PATH}/{layout}/{algo}/s2"
            os.makedirs(f"{algo_s2_dir}/{exp}", exist_ok=True)
            os.system(f"mv {tmp_dir}/{policy_name}/actor_periodic_{version}.pt {algo_s2_dir}/{exp}/{seed}.pt")
            logger.success(f"{layout} {algo} {exp} {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract S2 models")
    parser.add_argument("-l", "--layout", type=str, help="layout name")
    parser.add_argument("-e", "--env", type=str, help="env name")
    parser.add_argument("-a", "--algo", "--algorithm", type=str, action="append", required=True)
    parser.add_argument("-p", type=float, help="percentile", default=0.8)
    parser.add_argument("-f", "--final",  action="store_true", help="select final checkpoint", default=False)

    args = parser.parse_args()
    layout = args.layout
    #assert layout in [
    #    "random0",
    #    "academy_3_vs_1_with_keeper",
    #    "random0_medium",
    #    "random1",
    #    "random3",
    #    "small_corridor",
    #    "unident_s",
    #    "random0_m",
    #    "random1_m",
    #    "random3_m",
    #    "academy_3_vs_1_with_keeper",
    #    "inverse_marshmallow_experiment",
    #    "subobjective",
    #    "random3_l_m",
    #    "forced_coordination_tomato",
    #    "placement_coordination",
    #    "all",
    #]
    if layout == "all":
        layout = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        layout = [layout]
    algorithms = args.algo
    percentile = args.p

    assert all([algo in ["traj", "mep", "fcp", "cole", "hsp"] for algo in algorithms])
    ALG_EXPS = {
        "fcp": [
            "fcp-S2-s24",
            "fcp-S2-s36",
        ],
        "mep": [
            "mep-S2-s10",
            "mep-S2-s24",
            "mep-S2-s36",
        ],
        "hsp": [

            #"hsp-S2-s12",
            #"hsp-S2-s24",

            #"primitive_hsp_plate_placement-S2-s5",
            #"hsp_plate_placement_shared-S2-s5",
            
            "primitive_hsp_onion_tomato-S2-s10",
            "hsp_onion_tomato_shared-S2-s10"

            #"hsp_plate_placement_shared-S2-s10",
            #"reactive_hsp_plate_placement_shared-S3-s10",
            #"reactive2_hsp_plate_placement_shared-S3-s10",
        ],
        "traj": [
            "traj-S2-s24",
            "traj-S2-s36",
        ],
        "cole": ["cole-S2-s50", "cole-S2-s75"],
    }

    hostname = socket.gethostname()
    logger.info(f"hostname: {hostname}")
    for l in layout:
        for algo in algorithms:
            logger.info(f"for layout {l}")

            i = 0
            # for exp in ALG_EXPS[algo]:
            #     extract_pop_S2_models(l, algo, exp, args.env, percentile)
            while i < len(ALG_EXPS[algo]):
                exp = ALG_EXPS[algo][i]
                try:
                    extract_pop_S2_models(l, algo, exp, args.env, percentile, args.final)
                except Exception as e:
                    logger.error(e)
                else:
                    i += 1