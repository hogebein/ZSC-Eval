import argparse
import os
import os.path as osp

from loguru import logger

policy_pool_dir = "../policy_pool"

S1_POP_EXPS = {
    "hsp-S1": "hsp/s1/hsp_plate_shared",
    "mep-S2" : "mep/s2/mep-S2-s36",
    "hsp-S2" : "hsp/s2/hsp_plate-S2-s36",
}

PT_NUM = {
    "hsp-S1": 2,
    "mep-S2" : 1,
    "hsp-S2" : 1,
}

PT_CONFIG = {
    "hsp-S1": "mlp",
    "mep-S2" : "rnn",
    "hsp-S2" : "rnn",
}

ALG_NAME = {
    "hsp-S1": "hsp"
}

N_REPEAT = 30


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument("alg", type=str)
    parser.add_argument("total", type=int)
    parser.add_argument("pop", type=int)

    args = parser.parse_args()

    if args.layout == "all":
        layouts = [
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
        layouts = [args.layout]

    for layout in layouts:
        
        exp = S1_POP_EXPS[args.alg]
        source_dir = osp.join(policy_pool_dir, layout, exp)
        pt_lst = os.listdir(source_dir)
        logger.debug(pt_lst)
        pop_alg = args.alg if args.alg != "fcp" else "sp"
        #pt_lst.sort(key=lambda pt: int(pt.split("_", 1)[0][len(pop_alg) :]))
        pt_lst.sort()
        if args.alg == "fcp":
            pt_lst = pt_lst[: args.total * PT_NUM[args.alg]]
            logger.info(f"pop size {len(pt_lst)}: {pt_lst}")
        yml_dir = osp.join(
            policy_pool_dir,
            layout,
            "hsp/cp",
            exp.split("/")[-1]
        )
        # logger.debug(exp.split("/")[-1])
        os.makedirs(yml_dir, exist_ok=True)
        for n_r in range(N_REPEAT):
            yml_path = osp.join(
                yml_dir,
                f"train-s{args.pop * PT_NUM[args.alg]}-{n_r+1}.yml",
            )
            logger.info(f"Writing cross-play yml for {exp} seed {n_r} in {yml_path}")
            yml = open(
                yml_path,
                "w",
                encoding="utf-8",
            )
            yml.write(
                f"""\
hsp_cp:
    policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: True
"""
            )
            for p_i in range(1, args.pop + 1):
                pt_i = (args.total // N_REPEAT * n_r + p_i - 1) % args.total + 1
                logger.debug(pt_i)
                if PT_NUM[args.alg] == 2:
                    actor_names = [
                    #    f"{pop_alg}{pt_i}_init_actor.pt",
                        f"{ALG_NAME[pop_alg]}{pt_i}_mid_actor.pt",
                        f"{ALG_NAME[pop_alg]}{pt_i}_final_actor.pt",
                    ]

                    for actor_name in actor_names:
                        print(actor_name)
                        assert actor_name in pt_lst, (actor_name, pt_lst)
                    yml.write(
                        f"""\
hsp{p_i}_1:
    policy_config_path: {layout}/policy_config/{PT_CONFIG[args.alg]}_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, exp, actor_names[0])}
hsp{p_i}_2:
    policy_config_path: {layout}/policy_config/{PT_CONFIG[args.alg]}_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, exp, actor_names[1])}
"""
                    )
                    

                elif PT_NUM[args.alg] == 1:
                    actor_names = [
                        f"{pt_i}.pt",
                    ]

                    for actor_name in actor_names:
                        print(actor_name)
                        assert actor_name in pt_lst, (actor_name, pt_lst)
                    yml.write(
                        f"""\
hsp{p_i}:
    policy_config_path: {layout}/policy_config/{PT_CONFIG[args.alg]}_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {os.path.join(layout, exp, actor_names[0])}
"""
                    )

                
            yml.close()
