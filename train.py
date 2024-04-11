import os
import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.reside_dataloader import RESIDE_Dataset
from model.AECRNet import AECRNet
from model.ContrastiveRegularization import ContrastiveRegularization
from utils.constants import batch_size, crop_size, epochs, model_path, model_dir, eval_epoch
from utils.metrics import ssim, psnr
from utils.utils import set_seed


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion):
    start_time = time.time()
    start_epoch = 0
    losses = []
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    if os.path.exists(model_path):
        print(f"Continue training from {model_path}")
        loaded_model = torch.load(model_path)
        losses = loaded_model['losses']
        model.load_state_dict(loaded_model['model'])
        start_epoch = loaded_model['epoch']
        optimizer.load_state_dict(loaded_model['optimizer'])
        max_ssim = loaded_model['max_ssim']
        max_psnr = loaded_model['max_psnr']
        psnrs = loaded_model['psnrs']
        ssims = loaded_model['ssims']
        print(f'max_psnr: {max_psnr} | max_ssim: {max_ssim}')
        print(f'start_epoch: {start_epoch} | start training')
    else:
        print('Training from scratch')

    for epoch in range(start_epoch + 1, epochs):
        model.train()
        for image, label in tqdm(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            pred = model(image)

            loss_l1 = criterion[0](pred, label)
            loss_cr = criterion[1](pred, label, image)

            loss = loss_l1 + loss_cr

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            print(
                f'loss:{loss.item():.5f} | l1:{loss_l1:.5f} | cr: {loss_cr:.5f} | '
                f'time_elapsed :{(time.time() - start_time) / 60 :.1f}'
            )
            if epoch % eval_epoch == 0:
                ssim_eval, psnr_eval = test(model, test_dataloader)
                print(f'epoch: {epoch} | ssim: {ssim_eval:.4f}| psnr: {psnr_eval:.4f}')

                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)

                torch.save({
                    'epoch': epoch,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'losses': losses,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_path)
    print("Done training")


def test(model, test_dataloader):
    model.eval()
    ssims = []
    psnrs = []

    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            pred = model(inputs)
            ssim_eval = ssim(pred, targets).item()
            psnr_eval = psnr(pred, targets)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

    return np.mean(ssims), np.mean(psnrs)


if __name__ == '__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    set_seed(42)
    its_train_path = 'dataset/indoor-training-set-its-residestandard'
    its_test_path = 'dataset/synthetic-objective-testing-set-sots-reside/indoor'

    ITS_train_loader = DataLoader(dataset=RESIDE_Dataset(its_train_path, train=True, size=crop_size),
                                  batch_size=batch_size,
                                  shuffle=True)
    ITS_test_loader = DataLoader(dataset=RESIDE_Dataset(its_test_path, train=False, size='whole img'), batch_size=1,
                                 shuffle=False)

    dataloaders = {
        "ITS": {
            'train': ITS_train_loader,
            'test': ITS_test_loader,
        }
    }
    dataset = "ITS"
    train_dataloader = dataloaders[dataset]['train']
    test_dataloader = dataloaders[dataset]['test']

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # because my gpu doesnt have enough memory ;(
    learning_rate = 0.0001
    models = {'AECR_Net': AECRNet(3, 3)}
    network = 'AECR_Net'
    model = models[network]
    model.to(device)

    max_iterations = len(train_dataloader) * epochs
    print("max iterations: ", max_iterations)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    criterion = [nn.L1Loss().to(device), ContrastiveRegularization(device)]
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations)

    train_model(model, train_dataloader, test_dataloader, optimizer, criterion)
