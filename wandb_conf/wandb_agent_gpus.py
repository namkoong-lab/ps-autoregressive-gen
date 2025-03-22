import argparse
import subprocess
import torch

# Weights and Biases Agent

parser = argparse.ArgumentParser()
parser.add_argument("agent_name")
parser.add_argument("--ngpus", default=None, type=int)
parser.add_argument("--start_gpu", default=0, type=int)
parser.add_argument("--gpus", nargs='*', type=int)
args = parser.parse_args()
ngpus = torch.cuda.device_count()

assert args.gpus is None or args.ngpus is None

if args.gpus is not None:
    gpus = args.gpus
elif args.ngpus is not None:
    gpus = range(args.start_gpu, args.start_gpu+args.ngpus)
else:
    gpus = range(ngpus)
for i in gpus:
    s = 'CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES={} wandb agent {} &'.format(i, args.agent_name)
    subprocess.call(s, shell=True)

