import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import sys
sys.path.append('../..')
from dset_helpers import data_given_param
from train_VMC import Train_w_VMC
from train_data import Train_w_Data
from train_hybrid import Train_w_Data_then_VMC

#--
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Parser for starting multiple runs on Graham')
# Required positional argument
parser.add_argument('delta', type=float,
                    help='A required float argument: value of delta') ## dont need -- in front of it
# Required positional argument
parser.add_argument('data_epochs', type=int,
                    help='A required integer argument: number of data pre-training steps')
# Optional argument
parser.add_argument('--rnn_dim', type=str, default='OneD',
                    help='An optional string argument: dimension of rnn used')
# Optional argument
parser.add_argument('--nh', type=int, default=32,
                    help='An optional integer argument: number of hidden units')
# Optional argument
parser.add_argument('--seed', type=int, default=100,
                    help='An optional integer argument: seed for RNG')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='An optional float argument: learning rate')
args = parser.parse_args()
#--

Lx = 16
Ly = 16
Omega = 4.24
Rb = 1.15
V0 = Rb**6 * Omega
sweep_rate = 15

delta_arg = args.delta
data_steps_arg = args.data_epochs
rnn_dim_arg = args.rnn_dim
nh_arg = args.nh
seed_arg = args.seed
lr_arg = args.lr

def main():
    config = {
        'name': 'Figure2', # A very random name for each experiment

        'Lx':Lx,  # number of sites in x-direction                    
        'Ly':Ly,  # number of sites in the y-directioni
        'V': V0,
        'Omega': Omega,
        'delta': delta_arg,
        'sweep_rate':sweep_rate,
        
        'nh': nh_arg,  # number of memory/hidden units
        'lr': lr_arg,  # learning rate
        'weight_sharing': True,
        'trunc': 100,
        'seed': seed_arg,
        
        'RNN': rnn_dim_arg,
        'VMC_epochs':2000,
        'Data_epochs':data_steps_arg,
        
        'ns': 100,
        'batch_size_data': 100,
        
        'Print':True,
        'Write_Data': True,
        'CKPT':True
        }
    
    return Train_w_Data_then_VMC(config)


if __name__ == "__main__":
    model,e,v,sm,c = main()
