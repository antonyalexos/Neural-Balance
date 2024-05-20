import multiprocessing
import subprocess

GPU = '1'

def run_script(command):
    subprocess.run(command, check=True)

commands = []

for model in ['large_fcn']:
    for fbAtStart in ['0']:
        for seed in [10, 11, 12, 13, 14]:
            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/MnistFC/Mnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--epochs')
            command.append('1000')
            command.append('--l2_weight')
            command.append('.0001')
            command.append('--lr')
            command.append('.0001')
            command.append('--seed')
            command.append(str(seed+5))
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            commands.append(command)

            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/MnistFC/Mnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--epochs')
            command.append('1000')
            command.append('--lr')
            command.append('.0001')
            command.append('--seed')
            command.append(str(seed+10))
            command.append('--neural_balance')
            command.append('1')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            commands.append(command)

            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/MnistFC/Mnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--epochs')
            command.append('1000')
            command.append('--lr')
            command.append('.0001')
            command.append('--seed')
            command.append(str(seed+15))
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            commands.append(command)

            command = []
            command.append('/bin/python3')
            command.append('/baldig/proteomics2/ian/Neural-Balance/personalExps/MnistFC/Mnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append(GPU)
            command.append('--epochs')
            command.append('1000')
            command.append('--l2_weight')
            command.append('.0001')
            command.append('--lr')
            command.append('.0001')
            command.append('--seed')
            command.append(str(seed+20))
            command.append('--neural_balance')
            command.append('1')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            commands.append(command)

print(len(commands))

# Create a pool of workers and map commands to the workers
with multiprocessing.Pool(processes=20) as pool:
    pool.map(run_script, commands)