import multiprocessing
import subprocess

GPU = '0'
lr = '.0001'

def run_script(command):
    subprocess.run(command, check=True)

commands = []

for model in ['3']:
    for fbAtStart in ['0']:
        for seed in ['5']:
            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/IMDbRnn/rnn_imdb.py')
            command.append('--n_layers')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--l2_weight')
            command.append('.0001')
            command.append('--lr')
            command.append(lr)
            command.append('--seed')
            command.append(seed)
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--trainDataFrac')
            command.append('.05')
            commands.append(command)

            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/IMDbRnn/rnn_imdb.py')
            command.append('--n_layers')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--lr')
            command.append(lr)
            command.append('--seed')
            command.append(seed)
            command.append('--neural_balance')
            command.append('1')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--trainDataFrac')
            command.append('.05')
            commands.append(command)

            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/IMDbRnn/rnn_imdb.py')
            command.append('--n_layers')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--lr')
            command.append(lr)
            command.append('--seed')
            command.append(seed)
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--trainDataFrac')
            command.append('.05')
            commands.append(command)

print(len(commands))

# Create a pool of workers and map commands to the workers
with multiprocessing.Pool(processes=3) as pool:
    pool.map(run_script, commands)