import torch
import torchaudio
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
from tqdm import tqdm

from dataset import Denoise_data
from data_augment import *
from model import Demucs

import warnings
warnings.filterwarnings("ignore")

def val(args, model, val_loader):
    L1_loss = torch.nn.L1Loss().cuda()
    mrstftloss = MultiResolutionSTFTLoss(factor_sc=0.5, factor_mag=0.5).cuda()
    model.eval()

    total_loss = 0
    pesq_list = []
    with torch.no_grad():
        for noise, clean in val_loader:
            noise, clean = noise.cuda(), clean.cuda()
            pred = model(noise)

            # Loss calculation
            loss = L1_loss(clean, pred)
            sc_loss, mag_loss = mrstftloss(pred.squeeze(1), clean.squeeze(1))
            loss += sc_loss + mag_loss
            
            pred = pred.cpu().detach()
            clean = clean.cpu().detach()

            total_loss += loss.item()
            pesq = pesq_cal(pred, clean)
            pesq_list.append(pesq)

    pesq_list = np.array(pesq_list)
    mean_pesq = np.mean(pesq_list)
    print('loss:%.4f \t mean_pesq:%.4f' % (total_loss, mean_pesq))


def test(args, model, test_loader):
    model.eval()

    with torch.no_grad():
        for noise, audio_name in tqdm(test_loader):
            num = audio_name[0].split('_')[1]
            noise = noise.cuda()
            pred = model(noise)

            pred = pred.cpu().detach().squeeze(0)

            save_path = os.path.join(args.output_path, 'vocal_%s.flac' % num)
            torchaudio.save(save_path, pred, args.sample_rate)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../dataset', help="training image path")
    parser.add_argument("--output_path", default='./output', help="Path to output audio")
    parser.add_argument('--mode', default='test', help='val/test')

    # Data processing
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sampling rate of the audio")

    # Model parameters
    parser.add_argument("--hidden", type=int, default=48, help="Hidden size of the model")
    parser.add_argument("--depth", type=int, default=5, help="Depth of the model")
    parser.add_argument("--kernel_size", type=int, default=8, help="Kernel size of the model")

    parser.add_argument('--load', default='./model_best.pth', help='path to model to continue training')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu workers')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    data = Denoise_data(args.data_path, args.mode)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.n_cpu, drop_last=False)

    model = Demucs(args).cuda()
    model.load_state_dict(torch.load(args.load))

    if args.mode == 'val':
        val(args, model, dataloader)
    elif args.mode == 'test':
        test(args, model, dataloader)




