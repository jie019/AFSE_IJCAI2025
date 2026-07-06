from abc import abstractmethod
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataset import *
from metric import *
from loss import *
from rich.progress import Progress
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import time
import os
from torch.utils.data import DataLoader, DistributedSampler


class SortedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        # 对数据集进行全局排序
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i]['noisy_speech']))

        # 初始化父类
        super(SortedDistributedSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
                                                       seed=seed)

    def __iter__(self):
        # 获取父类分配的 indices
        indices = list(super().__iter__())

        # 返回排序后索引中的对应部分
        return iter([self.sorted_indices[i] for i in indices])


class BasicTrainer():
    def __init__(self, train_data, valid_data, model, logger, opt):
        # dataset
        self.train_data = train_data
        self.valid_data = valid_data
        self.device=opt.device
        collate = CustomCollate(opt)
        # data
        self.train_loader = DataLoader(self.train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        self.valid_loader = DataLoader(self.valid_data, batch_size=opt.batch_size, shuffle=False, drop_last=True,
                                       pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        # model
        self.model = model.to(self.device)
        self.log_output_path='./asset/VCTK'
        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.K=65536
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

    def save_cpt(self, step, save_path):
        """
        save checkpoint, for inference/re-training
        :return:
        """
        torch.save(
            {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict()
            },
            save_path
        )
    def save_cpt1(self, step, save_path):
        """
        save checkpoint, for inference/re-training
        :return:
        """
        torch.save(
            {
                'step': step,
                'model_state_dict': self.G_model.state_dict(),
                'optimizer_state_dict': self.G_optim.state_dict()
            },
            save_path
        )

class VBTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, model, logger, opt):
        super(VBTrainer, self).__init__(train_data, valid_data, model, logger, opt)

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

    def train(self):

        if not os.path.exists(self.log_output_path):
            os.makedirs(self.log_output_path)
        best_cv_loss = float("inf")
        harving = False
        with open(os.path.join(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'batchsize:{self.opt.batch_size} \n')
        for epoch in range(self.opt.n_epoch):
            self.model.train()
            epoch_time = time.time()
            losses = []
            for i,batch in enumerate(tqdm(self.train_loader)):
                # cuda
                for key in batch.keys():
                    try:
                        batch[key] = batch[key].to(self.device)
                    except AttributeError:
                        continue

                out = self.run_step(batch)
                self.optim.zero_grad()
                out['loss'].backward()
                losses.append(out['loss'].item())
                self.optim.step()
                if self.opt.wandb:
                    wandb.log(
                        {
                            'train_loss': out['loss'].item()
                        }
                    )
            print("Epoch: {} cost time: {} train_loss :{}".format(epoch + 1, time.time() - epoch_time,np.mean(losses)))
            with open(os.path.join(self.log_output_path, 'log.txt'), 'a') as f:
                f.write(f'Epoch:{epoch} | Time:{time.time() - epoch_time} | Train Loss:{np.mean(losses)} ')
            mean_valid_loss = self.inference()

            '''Adjust the learning rate and early stop'''


            if harving == True:
                optim_state = self.optim.state_dict()
                for i in range(len(optim_state['param_groups'])):
                    optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                self.optim.load_state_dict(optim_state)
                self.logger.print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                harving = False

            if mean_valid_loss < best_cv_loss:
                self.logger.print(
                    f"best loss is: {best_cv_loss}, current loss is: {mean_valid_loss}, save best_checkpoint.pth")
                best_cv_loss = mean_valid_loss

                self.save_cpt(epoch,
                              save_path=os.path.join(self.log_output_path,
                                        f'{self.model.__class__.__name__}_best.pth'))
            self.save_cpt(epoch,
                          save_path=os.path.join(self.log_output_path,
                                    f'{self.model.__class__.__name__}_{epoch}.pth'))

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        if self.progress:
            loss = self.inference_()
        else:
            loss = self.inference_()
        return loss

    def inference_(self):
        self.model.eval()
        loss_list = []
        csig_list, cbak_list, covl_list, pesq_list,wbpesq_list,nbpesq_list, ssnr_list, stoi_list,snr_list,sisnr_list= [], [], [], [], [], [],[],[],[],[]
        for i,batch in enumerate(tqdm(self.valid_loader)):
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.device)
                except AttributeError:
                    continue

            out = self.run_step(batch)  # out['compressed_feats']
            batch_result = compare_complex(out['model_out']['est_comp'], out['compressed_label'],
                                           batch['frame_num_list'],batch['c_list'],batch['c_list1'],
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
            f.write(f'Test_LOSS:{np.mean(loss_list):.4f} | Test PESQ:{np.mean(pesq_list):.3f} | Test WBPESQ:{np.mean(wbpesq_list):.3f} | Test NBPESQ:{np.mean(nbpesq_list):.3f}' 
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
                    'test_mean_snr':np.mean(snr_list),
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
