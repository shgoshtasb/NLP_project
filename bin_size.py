import numpy as np
import os, sys
import subprocess
task = 'DailyDialog'
num_updates = 5
calloss_lambda = 0.6
learning_rate = 2e-5

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bin_size", type=int, default=10)

args = parser.parse_args()
bin_size = args.bin_size

output_dir = task.lower() + '_bin{}'.format(bin_size)
task_name = task.lower()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
cmd = 'python code/classify_bert.py --model_type bert --model_name_or_path bert-base-uncased --task_name {task_name} --do_train --data_dir xslue_data/{task} --output_dir {output_dir} --num_updates {num_updates} --learning_rate {learning_rate} --bin_size {bin_size} --calloss_lambda 0.0 --num_train_epochs 30 --num_train_epochs 30 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32'
process = subprocess.Popen(cmd.format(task_name=task_name, task=task, output_dir=output_dir, num_updates=num_updates, learning_rate=learning_rate, bin_size=bin_size, calloss_lambda=calloss_lambda), shell=True, stdout=subprocess.PIPE)
for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
    print(line)

output_dir = task.lower() + '_bin{}'.format(bin_size) + '_poscal'
task_name = task.lower()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
cmd = 'python code/classify_bert.py --model_type bert --model_name_or_path bert-base-uncased --task_name {task_name} --do_train --data_dir xslue_data/{task} --output_dir {output_dir} --num_updates {num_updates} --learning_rate {learning_rate} --bin_size {bin_size} --calloss_lambda {calloss_lambda} --num_train_epochs 30 --poscal_train  --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32'
process = subprocess.Popen(cmd.format(task_name=task_name, task=task, output_dir=output_dir, num_updates=num_updates, learning_rate=learning_rate, bin_size=bin_size, calloss_lambda=calloss_lambda), shell=True, stdout=subprocess.PIPE)
for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
    print(line)

output_dir = task.lower() + '_bin{}'.format(bin_size) + '_plattbin'
task_name = task.lower()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
cmd = 'python code/classify_bert.py --model_type bert --model_name_or_path bert-base-uncased --task_name {task_name} --do_train --data_dir xslue_data/{task} --output_dir {output_dir} --num_updates {num_updates} --learning_rate {learning_rate} --bin_size {bin_size} --calloss_lambda {calloss_lambda} --num_train_epochs 30 --plattbin_train  --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32'
process = subprocess.Popen(cmd.format(task_name=task_name, task=task, output_dir=output_dir, num_updates=num_updates, learning_rate=learning_rate, bin_size=bin_size, calloss_lambda=calloss_lambda), shell=True, stdout=subprocess.PIPE)
for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
    print(line)

