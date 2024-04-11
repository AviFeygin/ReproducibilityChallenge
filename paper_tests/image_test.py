import torch
import torchvision.transforms as T
import torchvision.transforms as tfs
from PIL import Image

from model.AECRNet import AECRNet
from utils.constants import model_path

if __name__ == "__main__":
    model = AECRNet(3, 3)
    device = torch.device('cpu')  # because my gpu doesnt have enough memory ;(

    loaded_model = torch.load(f"../{model_path}")
    losses = loaded_model['losses']
    model.load_state_dict(loaded_model['model'])
    start_epoch = loaded_model['epoch']
    # optimizer.load_state_dict(loaded_model['optimizer'])
    max_ssim = loaded_model['max_ssim']
    max_psnr = loaded_model['max_psnr']
    psnrs = loaded_model['psnrs']
    ssims = loaded_model['ssims']
    print(f'max_psnr: {max_psnr} | max_ssim: {max_ssim}')
    print(f'start_epoch: {start_epoch} | start training')


    # Copied from https://discuss.pytorch.org/t/why-do-images-look-weird-after-imagenet-normalization/92071/3
    def renormalize(tensor):
        minFrom = tensor.min()
        maxFrom = tensor.max()
        minTo = 0
        maxTo = 1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


    img_dir = "../dataset/synthetic-objective-testing-set-sots-reside"
    clear = Image.open(img_dir + "/indoor/clear/1415.png")
    haze = Image.open(img_dir + "/indoor/hazy/1415_10.png")

    # clear.show()
    # haze.show()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_to_image = T.ToPILImage()
    model.eval()
    data = tfs.ToTensor()(haze)
    data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
    print(data.shape)
    data = data.unsqueeze(0)
    print(data.shape)
    with torch.no_grad():
        pred = model(data)
        print(pred.shape)
        pred = pred.squeeze()
        pred = renormalize(pred)
        pred_image = transform_to_image(pred)
        pred_image.show()
