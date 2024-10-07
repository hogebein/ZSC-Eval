import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.population.policy_pool import PolicyPool
from zsceval.algorithms.population.trainer_pool import TrainerPool
from zsceval.algorithms.population.utils import _t2n

from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

from zsceval.algorithms.utils.util import check
from zsceval.utils.util import get_gard_norm, huber_loss, mse_loss
from zsceval.utils.valuenorm import ValueNorm

class R_MAPPO_Trainer(TrainerPool):
    def __init__(self, args, policy_pool: PolicyPool, device=torch.device("cpu")):
        super(R_MAPPO_Trainer, self).__init__(args, policy_pool, device)

        self.stage = args.stage
        self.num_mini_batch = args.num_mini_batch
        self.share_policy = args.share_policy
        self.eval_policy = args.eval_policy
        self.num_agents = args.num_agents
        self.args = args


    def init_population(self):
        super(R_MAPPO_Trainer, self).init_population()

        self.agent_name = self.all_args.adaptive_agent_name
        self.population = {
            trainer_name: self.trainer_pool[trainer_name]
            for trainer_name in self.trainer_pool.keys()
            if self.agent_name not in trainer_name
        }
        if self.eval_policy != "":
            self.population = {
                trainer_name: self.population[trainer_name]
                for trainer_name in self.population.keys()
                if self.eval_policy not in trainer_name
            }
        self.population_size = self.all_args.population_size

        # print(self.population.keys(), self.population_size, self.stage)
        logger.info(f"population keys {self.population.keys()}, size {self.population_size}, stage {self.stage}")

        if self.share_policy:
            assert len(self.population) == self.population_size, len(self.population)
        else:
            assert len(self.population) == self.population_size * self.num_agents, (
                len(self.population),
                self.population_size,
                self.num_agents,
            )
            all_trainer_names = self.trainer_pool.keys()
            all_trainer_names = [x[: x.rfind("_")] for x in all_trainer_names]
            for a in range(self.num_agents):
                for x in all_trainer_names:
                    assert f"{x}_{a}" in self.trainer_pool.keys()

    def reward_shaping_steps(self):
        reward_shaping_steps = super(R_MAPPO_Trainer, self).reward_shaping_steps()
        if self.stage == 1:
            return [x // 2 for x in reward_shaping_steps]
        return reward_shaping_steps

    def save_steps(self):
        steps = super().save_steps()
        # logger.info(f"steps {steps}")
        if self.stage == 1:
            steps = {trainer_name: v // 2 for trainer_name, v in steps.items()}
        return steps
    
