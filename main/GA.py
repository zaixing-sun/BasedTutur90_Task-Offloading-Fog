"""
This script demonstrates how to run the NPGAPolicy.

"""

import os
import sys
from multiprocessing import Pool, cpu_count

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from tqdm import tqdm
import yaml

from core.env import Env
from core.task import Task
from core.vis import *
from core.vis.vis_stats import VisStats
from core.vis.logger import Logger
from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency
from policies.npga.npga_policy import Individual, NPGAPolicy
from policies.npga.nsga_policy import NSGA2Policy

import numpy as np

def error_handler(error: Exception):
    """Customized error handler for different types of errors."""
    errors = ['DuplicateTaskIdError', 'NetworkXNoPathError', 'IsolatedWirelessNode', 'NetCongestionError', 'InsufficientBufferError']
    message = error.args[0][0]
    if message in errors:
        pass
    else:
        raise
    
def evaluate_individual(args):
    """
    Evaluate an individual solution.
    """
    
    m1 = SuccessRate()
    m2 = AvgLatency()
    
    policy, data, config = args
    
    env = create_env(config)
    
    until = 0
    launched_task_cnt = 0
    
    iter_data = data.iterrows()
    
    for i, task_info in iter_data:
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL']/10,
                    src_name='e0',
                    task_name=task_info['TaskName'])

        while True:
            # Catch completed task information.
            while env.done_task_info:
                item = env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                dst_id, state = policy.act(env, task)  # offloading decision
                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1
                break

            # Execute the simulation with error handler.
            until += env.refresh_rate
            try:
                env.run(until=until)
            except Exception as e:
                error_handler(e)



    # Continue the simulation until the last task successes/fails.
    while env.task_count < launched_task_cnt:
        until += env.refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            pass
            
    success_rate = m1.eval(env.logger)
    avg_latency = m2.eval(env.logger)
    avg_power = env.avg_node_power()
    
    return success_rate, avg_latency, avg_power

def run_epoch(config, policy, data: pd.DataFrame, train=True):
    
    
    pool = Pool(processes=cpu_count())
    args = [(ind,  data, config) for ind in policy.individuals()]
    
    fitness = pool.map(evaluate_individual, args,)
    
    pool.close()
    pool.join()
    
    fitness = []
    
    for i in range(len(args)):
        fitness.append(evaluate_individual(args[i]))
        print(fitness[-1])

    
    SR, L, E = policy.best_individual(fitness)

    if train:
        policy.update(fitness)

    
    return SR, L, E


def create_env(config):
    """Create and return an environment instance."""
    dataset = config["env"]["dataset"]
    flag = config["env"]["flag"]
    scenario = Scenario(config_file=f"eval/benchmarks/{dataset}/data/{flag}/config.json", flag=flag)
    env = Env(scenario, config_file="core/configs/env_config_null.json", verbose=False)
    env.refresh_rate = config['env']['refresh_rate']
    return env



def main():

    config_path = "main/configs/GA/NSGA2.yaml"
    
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logger = Logger(config)
    

    env = create_env(config)
    
    # Load train and test datasets.
    train_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{config['env']['flag']}/trainset.csv")
    test_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{config['env']['flag']}/testset.csv")

    if config["policy"] == "NPGA":
        policy = NPGAPolicy(env, config)
    if config["policy"] == "NSGA2":
        policy = NSGA2Policy(env, config)
    
    m1 = SuccessRate()
    m2 = AvgLatency()
    
    
    for epoch in range(config["training"]["num_epochs"]):
        
        logger.update_epoch(epoch)

        # Training phase.
        
        logger.update_mode('Training')
        
        (SR, L, E) = run_epoch(config, policy, train_data, train=True)
        
        logger.update_metric('SuccessRate',SR)
        logger.update_metric('AvgLatency', L)
        logger.update_metric("AvgPower", E)
        
        env.close()
        
        # Testing phase.
        
        logger.update_mode('Testing')
        
        (SR, L, E) = run_epoch(config, policy, test_data, train=False)
        
        logger.update_metric('SuccessRate', SR)
        logger.update_metric('AvgLatency', L)
        logger.update_metric("AvgPower", E)
        
        env.close()


    logger.plot()
    logger.save_csv()
    
    vis_stats = VisStats(save_path=logger.log_dir)
    vis_stats.vis(env)
    
    logger.close()
    env.close()


if __name__ == '__main__':
    main()
