import argparse

from loguru import logger

from zsceval.utils.bias_agent_vars import LAYOUTS_EXPS, LAYOUTS_KS


def parse_args():
    parser = argparse.ArgumentParser(description="zsceval", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--layout", type=str, required=True, help="layout name")
    parser.add_argument("--eval_result_dir", type=str, default="eval/results")
    parser.add_argument("--policy_pool_path", type=str, default="../policy_pool")
    parser.add_argument("-a", "--algo", type=str, default="mep")
    parser.add_argument("-s", "--stage", type=str, default=2)
    parser.add_argument("-v", "--bias_agent_version", type=str, default="hsp")
    parser.add_argument("-t", "--training_type", type=str, default="s1")
    parser.add_argument("-k", "--K", type=str, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    layout = args.layout
    assert layout in list(LAYOUTS_EXPS.keys()) + ["all"]
    if layout == "all":
        layout = list(LAYOUTS_EXPS.keys())
    else:
        layout = [layout]
    policy_version = args.bias_agent_version

    for l in layout:
        logger.info(f"layout: {l}")
        s_exps = range(args.K)
        assert len(s_exps) == args.K

        # generate HSP evaluation config
        benchmark_yml_path = f"{args.policy_pool_path}/{l}/{args.algo}/s{args.stage}/{policy_version}/benchmarks-s{args.K}.yml"

        with open(
            benchmark_yml_path,
            "w",
            encoding="utf-8",
        ) as f:
            for i, exp_i in enumerate(s_exps):

                if args.stage == 1:
                    model_name = f"hsp{exp_i+1}_final_actor"
                else:
                    model_name = f"{exp_i+1}"



#                f.write(
#                    f"""\
#bias{i+1}_mid:
#    policy_config_path: {l}/policy_config/mlp_policy_config.pkl
#    featurize_type: ppo
#    train: False
#    model_path:
#        actor: {l}/hsp/s1/{policy_version}/hsp{exp_i+1}_mid_actor.pt\n"""
#                )
                f.write(
                    f"""\
bias{i+1}_final:
    policy_config_path: {l}/policy_config/rnn_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {l}/{args.algo}/s{args.stage}/{policy_version}/{model_name}.pt\n"""
                )
            f.write(
                f"""\
agent_name:
    policy_config_path: {l}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: {l}/algorithm/{args.training_type}/population/seed.pt
    utility: @@@"""
            )
        logger.success(f"write to {benchmark_yml_path}")
    
