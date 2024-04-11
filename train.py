import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.reside_dataloader import RESIDE_Dataset
from model.AECRNet import AECRNet
from utils.metrics import ssim, psnr

if __name__ == '__main__':
    crop_size = 240
    batch_size = 16

    its_train_path = 'dataset/indoor-training-set-its-residestandard'
    its_test_path = 'dataset/synthetic-objective-testing-set-sots-reside/indoor'

    ITS_train_loader = DataLoader(dataset=RESIDE_Dataset(its_train_path, train=True, size=crop_size),
                                  batch_size=batch_size,
                                  shuffle=True)
    ITS_test_loader = DataLoader(dataset=RESIDE_Dataset(its_test_path, train=False, size='whole img'), batch_size=1,
                                 shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # because my gpu doesnt have enough memory ;(
    learning_rate = 0.0001

    model = AECRNet(3, 3)
    model.to(device)
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08)

    for epoch in range(100):
        model.train()
        for image, label in tqdm(ITS_train_loader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            print('image', image.shape)
            print('label', label.shape)
            output = model(image)
            print(output.shape)
            break
        break

    model.eval()
    ssims, psnrs = [], []
    for i, (image, label) in enumerate(ITS_test_loader):
        image = image.to(device)
        label = label.to(device)
        print('inputs', image.shape)
        print('target', label.shape)
        with torch.no_grad():
            pred = model(image)
            ssim_val = ssim(pred, label).item()
            psnr_val = psnr(pred, label)
            ssims.append(ssim_val)
            psnrs.append(psnr_val)
        break
    print(ssims, psnrs)
