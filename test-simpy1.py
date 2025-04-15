import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 参数设置
# -----------------------
NUM_TASKS = 10
LAMBDA = 1.5
SEED = 42
np.random.seed(SEED)

# -----------------------
# 提前生成任务数据
# -----------------------
arrival_times = np.round(np.cumsum(np.random.exponential(1 / LAMBDA, NUM_TASKS)), 2)
execution_times = [
    (round(np.random.uniform(3, 7), 2), round(np.random.uniform(4, 9), 2))
    for _ in range(NUM_TASKS)
]

task_data = [{
    'TaskID': i,
    'Arrival': arrival_times[i],
    'ExecM1': execution_times[i][0],
    'ExecM2': execution_times[i][1]
} for i in range(NUM_TASKS)]

# -----------------------
# 初始化环境
# -----------------------
env = simpy.Environment()
machine_queues = [[], []]  # 每台机器的任务队列
records = []

# -----------------------
# 任务执行逻辑
# -----------------------
def task(env, tid, arrival, exec_m1, exec_m2):
    chosen_id = 0 if exec_m1 <= exec_m2 else 1
    exec_time = exec_m1 if chosen_id == 0 else exec_m2
    queue = machine_queues[chosen_id]

    my_event = env.event()
    queue.append((tid, exec_time, my_event))
    if len(queue) > 1:
        yield my_event  # 不是队首要等待

    start = env.now
    yield env.timeout(exec_time)
    finish = env.now

    records.append({
        'Task': f'Task-{tid}',
        'Arrival': arrival,
        'Start': round(start, 2),
        'Finish': round(finish, 2),
        'Wait': round(start - arrival, 2),
        'Duration': exec_time,
        'Machine': f'Machine-{chosen_id + 1}'
    })

    queue.pop(0)
    if queue:
        queue[0][2].succeed()  # 唤醒下一个

# -----------------------
# 主控逻辑（手动推进时间 + 提交任务）
# -----------------------
for task_info in task_data:
    arrival = task_info['Arrival']
    if env.now < arrival:
        env.run(until=arrival)
    env.process(task(env, task_info['TaskID'], arrival, task_info['ExecM1'], task_info['ExecM2']))

# 运行剩余任务
env.run()

# -----------------------
# 输出和可视化
# -----------------------
df = pd.DataFrame(records).sort_values(by='Start').reset_index(drop=True)

# Gantt 图
fig, ax = plt.subplots(figsize=(10, 4))
colors = {'Machine-1': 'skyblue', 'Machine-2': 'lightgreen'}
for _, row in df.iterrows():
    ax.barh(row['Machine'], row['Finish'] - row['Start'],
            left=row['Start'], color=colors[row['Machine']], edgecolor='black')
    ax.text(row['Start'] + 0.1, row['Machine'], row['Task'],
            va='center', ha='left', fontsize=8)

ax.set_xlabel("Time")
ax.set_title("Task Execution Gantt Chart")
ax.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 打印调度记录
print(df)
