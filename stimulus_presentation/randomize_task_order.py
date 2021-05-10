import numpy as np

tasks = [
    "Expectation\n",
    "Resting state\nDRMT-session\nDRMT resting state",
    "Kornhuber Task\n"
]

selection = np.arange(len(tasks))

np.random.shuffle(selection)

for i, sel in enumerate(selection):
    print(f'Condition {i+1}:\n{tasks[sel]}')

