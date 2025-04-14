"""
This script demonstrates how to use the Topo4MEC dataset.
"""

import os
import sys
import random
import numpy as np

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from core.env import Env_Zaixing as Env
from core.task import Task
from core.vis import *
from eval.benchmarks.Topo4MEC.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency  # metric
from policies.demo.demo_random import DemoRandom  # policy
from policies.demo.demo_greedy import GreedyPolicy
from policies.demo.demo_round_robin import RoundRobinPolicy

def new_task(test_tasks):
    task_info = test_tasks.pop(0)
    generated_time = task_info[1]
    task = Task(task_id=task_info[2],
                task_size=task_info[3],
                cycles_per_bit=task_info[4],
                trans_bit_rate=task_info[5],
                ddl=task_info[6],
                src_name=task_info[7],
                task_name=task_info[0])
    return task, generated_time

def wrapped_task(env, task, dst_name, done_event):
    yield from env._execute_task (task, dst_name)
    # if env.task_count >= remaining['count']:
    #     done_event.succeed()  # 最后一个任务执行完毕，立即触发终止信号
    
    # 在任务完成时触发帧信息更新事件
    # env.controller.event().succeed()  # 触发帧信息更新
    # env.monitor_trigger.succeed()  # 任务执行完成，触发监控
    env.trigger_monitor()  # 改为调用安全触发方法


# ✅ 每个任务注册过程是 SimPy 协程
def task_submit_process(env, test_tasks, policy, done_event):
    env.total_task_count = len(test_tasks)  # 记录总任务数

    for task_info in test_tasks:
        generated_time = task_info[1]
        task = Task(task_id=task_info[2],
                    task_size=task_info[3],
                    cycles_per_bit=task_info[4],
                    trans_bit_rate=task_info[5],
                    ddl=task_info[6],
                    src_name=task_info[7],
                    task_name=task_info[0])

        wait = generated_time - env.now
        if wait > 0:
            yield env.controller.timeout(wait)

        dst_id = policy.act(env, task)
        dst_name = env.scenario.node_id2name[dst_id]
        # 提交任务时用包装函数，执行完自动减少计数
        env.controller.process(wrapped_task(env, task, dst_name, done_event))
        # env.monitor_trigger.succeed()  # 新任务提交，触发监控
        env.trigger_monitor()  # 改为调用安全触发方法


def main():
    flag = '25N50E'
    # flag = '50N50E'
    # flag = '100N150E'
    # flag = 'MilanCityCenter'

    # Create the environment with the specified scenario and configuration files.
    scenario = Scenario(config_file=f"eval/benchmarks/Topo4MEC/data/{flag}/config.json", flag=flag)
    env = Env(scenario, config_file="core/configs/env_config_null.json")

    # Load the test dataset.
    data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/{flag}/testset.csv")
    test_tasks = list(data.iloc[:].values)

    # Init the policy.
    random.seed(8783578275289)
    np.random.seed(878357)
    policy = DemoRandom()
    # policy = RoundRobinPolicy()
    # policy = GreedyPolicy()

    # 启动任务提交进程
    env.controller.process(task_submit_process(env, test_tasks, policy, env.done_event))
    
    # 运行仿真直到所有任务完成
    env.controller.run(until=env.done_event)
    

    # Begin the simulation.
    # until = 0
    # launched_task_cnt = 0
    # temp_time = 0
    # while test_tasks!= []:
    #     task, generated_time = new_task(test_tasks)
    #     yield env.controller.timeout(generated_time)
    #     dst_id = policy.act(env, task)  # offloading decision
    #     dst_name = env.scenario.node_id2name[dst_id]
    #     env.process(task=task, dst_name=dst_name)
    #     launched_task_cnt += 1

    # env.run()

    # # flag = True  # True: reactive, False: proactive
    # flag = False  # True: reactive, False: proactive
    # for task_info in test_tasks:
    #     # Task properties:
    #     # ['TaskName', 'GenerationTime', 'TaskID', 'TaskSize', 'CyclesPerBit', 
    #     #  'TransBitRate', 'DDL', 'SrcName', 'DstName']

    #     task, generated_time = new_task(launched_task_cnt, test_tasks)


    #     # generated_time = task_info[1]
    #     # task = Task(task_id=task_info[2],
    #     #             task_size=task_info[3],
    #     #             cycles_per_bit=task_info[4],
    #     #             trans_bit_rate=task_info[5],
    #     #             ddl=task_info[6],
    #     #             src_name=task_info[7],
    #     #             task_name=task_info[0])
        
    #     if flag:
    #         interval_time = generated_time - temp_time
    #         yield env.controller.timeout(interval_time)
    #         dst_id = policy.act(env, task)  # offloading decision
    #         dst_name = env.scenario.node_id2name[dst_id]
    #         env.process(task=task, dst_name=dst_name)
    #         launched_task_cnt += 1
    #         temp_time = generated_time

    #     else:
    #         while True:
    #             # Catch completed task information.
    #             while env.done_task_info:
    #                 item = env.done_task_info.pop(0)

    #             if env.now == generated_time:
    #                 dst_id = policy.act(env, task)  # offloading decision
    #                 dst_name = env.scenario.node_id2name[dst_id]
    #                 env.process(task=task, dst_name=dst_name)
    #                 launched_task_cnt += 1
    #                 break

    #             # Execute the simulation with error handler.
    #             try:
    #                 env.run(until=until)
    #             except Exception as e:
    #                 pass

    #             until += 1

    # # Continue the simulation until the last task successes/fails.
    # while env.task_count < launched_task_cnt:
    #     until += 1
    #     try:
    #         env.run(until=until)
    #     except Exception as e:
    #         pass

    # Evaluation
    print("\n===============================================")
    print("Evaluation:")
    print("===============================================\n")

    print("-----------------------------------------------")
    m1 = SuccessRate()
    r1 = m1.eval(env.logger)
    print(f"The success rate of all tasks: {r1:.4f}")
    print("-----------------------------------------------\n")

    print("-----------------------------------------------")
    m2 = AvgLatency()
    r2 = m2.eval(env.logger)
    print(f"The average latency per task: {r2:.4f}")

    print(f"The average energy consumption per node: {env.avg_node_energy():.4f}")
    print("-----------------------------------------------\n")

    env.close()


if __name__ == '__main__':
    main()
