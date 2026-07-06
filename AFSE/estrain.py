import argparse
from rich.console import Console
from estrainer import *
from utils.logger import *
from models.Base import Base
import os
import torch
import random
from dataset import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(opt):

    console = Console(color_system='256', style=None)
    if opt.wandb:
        wandb.init(project="dr_diffuse")
        # wandb.init(project="dr_diffuse", settings=wandb.Settings(start_method="fork"))
    else:
        print(console.print("wandb forbidden!"))

    # '''load data'''
    tr_data = VBDataset(
        '/home/wangshijie_qyh/VCTK_28spk_48k/train/noisy',
        '/home/wangshijie_qyh/VCTK_28spk_48k/train/clean',
        'train',
        opt)
    cv_data = VBDataset(
        '/home/wangshijie_qyh/VCTK_28spk_48k/test/noisy',
        '/home/wangshijie_qyh/VCTK_28spk_48k/test/clean',
        'valid',
        opt)


    console.print(f'total {tr_data.__len__()} train data, total {cv_data.__len__()} eval data.')
    opt.params = AttrDict(
        ours=False,
        fast_sampling=True,
        noise_schedule=np.linspace(1e-4, 0.05, 200).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
    )
    '''load model'''
    model = eval(opt.model)(opt)
    '''load trainer'''
    trainer = VBTrainer(tr_data, cv_data, model, console, opt)
    trainer.train()







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='manual seed')
    parser.add_argument('--model', type=str, default="Base", help='Base')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')#-3
    parser.add_argument('--n_epoch', type=int, default=70, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="weight decay")
    parser.add_argument('--half_lr', type=int, default=3, help='decay learning rate to half scale')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop training')
    parser.add_argument('--win_size', type=int, default=320)
    parser.add_argument('--fft_num', type=int, default=320)
    parser.add_argument('--win_shift', type=int, default=160)
    parser.add_argument('--chunk_length', type=int, default=48000)
    parser.add_argument('--feat_type', type=str, default='sqrt', help='normal/sqrt/cubic/log_1x')
    parser.add_argument('--wandb', action='store_true', help='load wandb or not')
    parser.add_argument('--c_out', type=int, default=301, help='output size') #101
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--num_nodes', type=int, default=301, help='to create Graph')
    parser.add_argument('--subgraph_size', type=int, default=3, help='neighbors number')
    parser.add_argument('--tanhalpha', type=float, default=3, help='')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    # GCN
    parser.add_argument('--node_dim', type=int, default=301, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=3, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.05, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')
    parser.add_argument('--world_size', type=int, default=1, help='number of GPUs')
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    # print(f'workspace:{os.getcwd()}, training device:{args.device}')
    main(args)

