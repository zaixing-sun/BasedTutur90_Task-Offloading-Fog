import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 参数配置
NUM_TASKS = 10
LAMBDA = 1.5
np.random.seed(42)

env = simpy.Environment()
machine_queues = [[], []]  # 每台机器的事件队列
records = []
task_counter = {'count': 0}
arrival_times = []

# 单个任务执行逻辑
def task(env, tid, exec_time_m1, exec_time_m2):
    chosen_id = 0 if exec_time_m1 <= exec_time_m2 else 1
    exec_time = exec_time_m1 if chosen_id == 0 else exec_time_m2
    queue = machine_queues[chosen_id]
    my_event = env.event()
    queue.append((tid, exec_time, my_event))

    if len(queue) > 1:
        yield my_event

    start = env.now
    yield env.timeout(exec_time)
    finish = env.now

    records.append({
        'Task': f'Task-{tid}',
        'Arrival': round(arrival_times[tid], 2),
        'Start': round(start, 2),
        'Finish': round(finish, 2),
        'Wait': round(start - arrival_times[tid], 2),
        'Duration': exec_time,
        'Machine': f'Machine-{chosen_id + 1}'
    })

    queue.pop(0)
    if queue:
        queue[0][2].succeed()

# 任务生成器
def task_generator(env, num_tasks, lambda_rate):
    for _ in range(num_tasks):
        inter_arrival = np.random.exponential(1 / lambda_rate)
        yield env.timeout(inter_arrival)
        arrival_times.append(env.now)
        exec_time_m1 = round(np.random.uniform(3, 7), 2)
        exec_time_m2 = round(np.random.uniform(4, 9), 2)
        env.process(task(env, task_counter['count'], exec_time_m1, exec_time_m2))
        task_counter['count'] += 1

# 启动仿真
env.process(task_generator(env, NUM_TASKS, LAMBDA))
env.run()

# 数据整理
df = pd.DataFrame(records).sort_values(by='Arrival').reset_index(drop=True)

# 绘制 Gantt 图
fig, ax = plt.subplots(figsize=(10, 4))
colors = {'Machine-1': 'skyblue', 'Machine-2': 'lightgreen'}

for _, row in df.iterrows():
    ax.barh(row['Machine'], row['Finish'] - row['Start'],
            left=row['Start'], color=colors[row['Machine']], edgecolor='black')
    ax.text(row['Start'] + 0.1, row['Machine'], row['Task'], va='center', ha='left', fontsize=8)

ax.set_xlabel("Time")
ax.set_title("Task Execution Gantt Chart (Dynamic Task Generation)")
ax.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(df)
