import numpy as np
from copy import deepcopy

expectation_task = "Expectation\n",

tasks = np.array([
    "Resting state\n",
    "DRMT\n",
    "Kornhuber Task\n"
])
np.random.shuffle(tasks)

rest_idx = np.where(tasks=="Resting state\n")[0][0]
drmt_idx = np.where(tasks=="DRMT\n")[0][0]

while rest_idx > drmt_idx:
    np.random.shuffle(tasks)
    rest_idx = np.where(tasks=="Resting state\n")[0][0]
    drmt_idx = np.where(tasks=="DRMT\n")[0][0]


print(tasks)

second_tasks = deepcopy(tasks[::-1])
print(tasks)
rest_idx = np.where(second_tasks=="Resting state\n")[0][0]
drmt_idx = np.where(second_tasks=="DRMT\n")[0][0]
tmp = second_tasks[rest_idx]
second_tasks[rest_idx] = second_tasks[drmt_idx]
second_tasks[drmt_idx] = tmp

all_tasks = np.append(np.append(tasks, expectation_task), second_tasks)
print(all_tasks)
