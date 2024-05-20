import pickle

import multiprocessing
import subprocess

lr = '.0001'

def run_script(command):
    subprocess.run(command, check=True)

commands = []
for seed in ['10', '11', '12', '13', '14']:
    for n_layers in ['2', '7']:
        for l2 in ['0.0001', '0.00001']:
            for nbs in ['0', '1']:
                command = []
                command.append('/bin/python3')
                command.append('/baldig/proteomics2/ian/Neural-Balance/DFA/main.py')
                command.append('--n_layers')
                command.append(n_layers)
                command.append('--gpu')
                command.append('0')
                command.append('--weight_decay')
                command.append(l2)
                command.append('--lr')
                command.append(lr)
                command.append('--seed')
                command.append(seed)
                command.append('--neuralFullBalanceAtStart')
                command.append(nbs)
                commands.append(command)
        for nb in ['0', '1']:
            for nbs in ['0', '1']:
                command = []
                command.append('/bin/python3')
                command.append('/baldig/proteomics2/ian/Neural-Balance/DFA/main.py')
                command.append('--n_layers')
                command.append(n_layers)
                command.append('--gpu')
                command.append('1')
                command.append('--lr')
                command.append(lr)
                command.append('--seed')
                command.append(seed)
                command.append('--nb')
                command.append(nb)
                command.append('--neuralFullBalanceAtStart')
                command.append(nbs)
                commands.append(command)

print(len(commands))

# Create a pool of workers and map commands to the workers
with multiprocessing.Pool(processes=80) as pool:
    pool.map(run_script, commands)