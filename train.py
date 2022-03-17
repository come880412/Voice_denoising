import torch
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
import tqdm

from dataset import Denoise_data
from data_augment import *
from model import Demucs
from STFT import MultiResolutionSTFTLoss

def train(args, model, optimizer, data_augments, train_loader, val_loader):
    L1_loss = torch.nn.L1Loss().cuda()
    mrstftloss = MultiResolutionSTFTLoss(factor_sc=0.5, factor_mag=0.5).cuda()

    for epoch in range(args.n_epochs):
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")

        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.cuda(), clean.cuda()

            # Data augmentation
            sources = torch.stack([noisy - clean, clean])
            sources = data_augments(sources)
            noise, clean = sources
            noisy = noise + clean

            pred = model(noisy)

            # Loss calculation
            loss = L1_loss(clean, pred)
            sc_loss, mag_loss = mrstftloss(pred.squeeze(1), clean.squeeze(1))
            loss += sc_loss + mag_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            )
        val(args, model, val_loader, epoch)
        torch.save(model.state_dict(), os.path.join(args.saved_model, 'model_epoch%d.pth' % (epoch)))

def val(args, model, val_loader, epoch):
    L1_loss = torch.nn.L1Loss().cuda()
    mrstftloss = MultiResolutionSTFTLoss(factor_sc=0.5, factor_mag=0.5).cuda()
    model.eval()

    total_loss = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Val[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for noise, clean in val_loader:
            noise, clean = noise.cuda(), clean.cuda()
            pred = model(noise)

            # Loss calculation
            loss = L1_loss(clean, pred)
            sc_loss, mag_loss = mrstftloss(pred.squeeze(1), clean.squeeze(1))
            loss += sc_loss + mag_loss

            total_loss += loss.item()
            pbar.update()
            pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='../dataset', help="training image path")
    parser.add_argument("--saved_model", default='./checkpoints/demucs', help="path to save model")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--lr_decay_epoch", type=int, default=30, help="Start to decay epoch")

    # Data augmentation
    parser.add_argument("--remix", action='store_true', help="Whether to remix noise and clean")
    parser.add_argument("--bandmask", type=int, default=0, help="Whether to drop at most this fraction of freqs in mel scale")
    parser.add_argument("--shift", type=int, default=0, help="Whether to use shift data augmentation")
    parser.add_argument("--shift_same", default=False, help="Whether to shift noise and clean by the same amount")
    parser.add_argument("--revecho", type=int, default=0, help="Whether to add reverb like augment")

    # Data processing
    parser.add_argument("--stride", type=int, default=1, help="How much to stride between training examples")
    parser.add_argument("--segment", type=int, default=4, help="Start to decay epoch")
    parser.add_argument("--sample_rate", type=int, default=16000, help="The sampling rate of the audio")

    # Model parameters
    parser.add_argument("--hidden", type=int, default=48, help="Hidden size of the model")
    parser.add_argument("--depth", type=int, default=5, help="Depth of the model")
    parser.add_argument("--kernel_size", type=int, default=8, help="Kernel size of the model")

    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument('--load', default='', help='path to model to continue training')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu workers')
    args = parser.parse_args()

    os.makedirs(args.saved_model, exist_ok=True)

    stride = args.stride * args.sample_rate
    length = args.segment * args.sample_rate

    train_data = Denoise_data(args.data_path, 'train', stride, length, True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
    val_data = Denoise_data(args.data_path, 'val', stride, length, True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.n_cpu, drop_last=False)

    # data augment
    data_augments = []
    if args.remix:
        data_augments.append(Remix())
    if args.bandmask:
        data_augments.append(BandMask(args.bandmask, sample_rate=16000))
    if args.shift:
        data_augments.append(Shift(args.shift, args.shift_same))
    if args.revecho:
        data_augments.append(
            RevEcho(args.revecho))
    data_augments = torch.nn.Sequential(*data_augments)
    model = Demucs(args).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    train(args, model, optimizer, data_augments, train_loader, val_loader)