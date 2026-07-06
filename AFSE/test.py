from abc import abstractmethod
import numpy as np
import torch
from rich.console import Console
import wandb
from torch.utils.data import DataLoader
from dataset import *
from metric import *
from loss import *
from tqdm import tqdm
import warnings
from models import *
warnings.filterwarnings('ignore')
import time
import os
import argparse
from models.Base import Base

class BasicTrainer():
    def __init__(self,  valid_data, model, logger, opt):
        # dataset
        self.device = opt.device
        self.valid_data = valid_data
        collate = CustomCollate(opt)
        # data
        self.valid_loader = DataLoader(self.valid_data, batch_size=opt.batch_size,shuffle=False, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        # model
        self.model = model.to(self.device)
        self.log_output_path='./asset/infer'
        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # others
        self.opt = opt
        self.t=int(16000/opt.win_shift+1)
        self.n=int(opt.fft_num/2+1)
        self.logger = logger  # can be wandb, logging, or rich.console
        self.progress = None
    @abstractmethod
    def run_step(self, x):
        pass

    @abstractmethod
    def train(self):
        pass


class VBTrainer(BasicTrainer):
    def __init__(self, valid_data, model, logger, opt):
        super(VBTrainer, self).__init__( valid_data, model, logger, opt)
    def data_compress(self, x):
        batch_feat = x['feats']
        batch_label = x['labels']
        noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])
        clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])
        if self.opt.feat_type == 'normal':
            batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
        elif self.opt.feat_type == 'sqrt':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                torch.norm(batch_label, dim=1)) ** 0.5
        elif self.opt.feat_type == 'cubic':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                torch.norm(batch_label, dim=1)) ** 0.3
        elif self.opt.feat_type == 'log_1x':
            batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                      torch.log(torch.norm(batch_label, dim=1) + 1)
        if self.opt.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
            batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                     dim=1)
            batch_label = torch.stack((batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                                      dim=1)
        return batch_feat, batch_label


    def run_step(self, x):
        batch_feat, batch_label = self.data_compress(x)
        out = self.model(batch_feat)
        loss = com_mag_mse_loss(out['est_comp'], batch_label, x['frame_num_list'])
        return {
            'model_out': out,
            'loss': loss,
            'compressed_feats': batch_feat,
            'compressed_label': batch_label,
        }

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        # dir=''
        # checkpoint = torch.load(dir)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.progress:
            loss = self.inference_()
        else:
            loss = self.inference_()
        return loss

    def inference_(self):
        # self.G_model.eval()
        self.model.eval()
        loss_list = []
        csig_list, cbak_list, covl_list, pesq_list, wbpesq_list, nbpesq_list, ssnr_list, stoi_list, snr_list, sisnr_list = [], [], [], [], [], [], [], [], [], []
        for i, batch in enumerate(tqdm(self.valid_loader)):
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.device)
                except AttributeError:
                    continue
            out = self.run_step(batch)  # out['compressed_feats']
            batch_result = compare_complex(out['model_out']['est_comp'], out['compressed_label'],
                                           batch['frame_num_list'],
                                           feat_type=self.opt.feat_type)
            loss_list.append(out['loss'].item())
            csig_list.append(batch_result[0])
            cbak_list.append(batch_result[1])
            covl_list.append(batch_result[2])
            pesq_list.append(batch_result[3])
            wbpesq_list.append(batch_result[4])
            nbpesq_list.append(batch_result[5])
            ssnr_list.append(batch_result[6])
            stoi_list.append(batch_result[7])
            snr_list.append(batch_result[8])
            sisnr_list.append(batch_result[9])
        with open(os.path.join(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(
                f'Test_LOSS:{np.mean(loss_list):.4f} | Test PESQ:{np.mean(pesq_list):.3f} | Test WBPESQ:{np.mean(wbpesq_list):.3f} | Test NBPESQ:{np.mean(nbpesq_list):.3f}'
                f'| Test CSIG:{np.mean(csig_list):.3f} | Test CBAK:{np.mean(cbak_list):.3f}'
                f'| Test COVL:{np.mean(covl_list):.3f} | TEST SSNR:{np.mean(ssnr_list):.3f}'
                f'| TEST STOI:{np.mean(stoi_list):.5f} | TEST SNR:{np.mean(snr_list):.3f} | Test SISNR:{np.mean(sisnr_list):.3f} \n')
        if self.opt.wandb:
            wandb.log(
                {
                    'test_loss': np.mean(loss_list),
                    'test_mean_csig': np.mean(csig_list),
                    'test_mean_cbak': np.mean(cbak_list),
                    'test_mean_covl': np.mean(covl_list),
                    'test_mean_pesq': np.mean(pesq_list),
                    'test_mean_wbpesq': np.mean(wbpesq_list),
                    'test_mean_nbpesq': np.mean(nbpesq_list),
                    'test_mean_ssnr': np.mean(ssnr_list),
                    'test_mean_stoi': np.mean(stoi_list),
                    'test_mean_snr': np.mean(snr_list),
                    'test_mean_sisnr': np.mean(sisnr_list)
                }
            )
        else:
            print({
                'test_loss': np.mean(loss_list),
                'test_mean_csig': np.mean(csig_list),
                'test_mean_cbak': np.mean(cbak_list),
                'test_mean_covl': np.mean(covl_list),
                'test_mean_pesq': np.mean(pesq_list),
                'test_mean_wbpesq': np.mean(wbpesq_list),
                'test_mean_nbpesq': np.mean(nbpesq_list),
                'test_mean_ssnr': np.mean(ssnr_list),
                'test_mean_stoi': np.mean(stoi_list),
                'test_mean_snr': np.mean(snr_list),
                'test_mean_sisnr': np.mean(sisnr_list)
            })
        return np.mean(loss_list)

    def save_cpt(self, step, save_path):
        super().save_cpt(step, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--chunk_length', type=int, default=160000)
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
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    cv_data = VBDataset(
        'datasets/DNS/test/noisy',
        'datasets/DNS/test/clean',
        'valid',
        args)
    model = eval(args.model)(args)
    console = Console(color_system='256', style=None)
    trainer = VBTrainer(cv_data, model, console,args)
    trainer.inference()
