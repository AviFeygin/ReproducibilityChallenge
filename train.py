from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.reside_dataloader import RESIDE_Dataset

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

    for epoch in range(1):
        for image, label in tqdm(ITS_train_loader):
            print('image', image.shape)
            print('label', label.shape)
            break

    for i, (inputs, targets) in enumerate(ITS_test_loader):
        print('inputs', inputs.shape)
        print('target', targets.shape)
        break
