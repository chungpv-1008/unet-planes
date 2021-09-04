import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import os
from torchvision import transforms

from utils.data_loading import BasicDataset

from unet import UNet

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    width, height = full_img.size
    img = BasicDataset.preprocess_image(full_img, scale_factor)
    img = torch.as_tensor(img).float().contiguous().unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return full_mask.argmax(dim=0).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='weights/weight.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', help='Filenames of input images',
                        default='./images/Chicago_Airport_0_0_910_10071.png')
    parser.add_argument('--output', '-o', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed', default=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')

    return parser.parse_args()

def mask_to_image(mask: np.ndarray):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_file = args.input
    out_file = f'{os.path.splitext(in_file)[0]}_output{os.path.splitext(in_file)[1]}'

    net = UNet(n_channels=3, n_classes=2)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    # img = cv2.imread(in_file)
    img = Image.open(in_file)

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)

    img = np.asarray(img)[:, :, :3]
    img = img[:, :, ::-1]

    output = img.copy()
    colors = [[0, 0, 0], [0, 255, 0]]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                output[i, j] = np.array(colors[mask[i, j]])
    alpha = 0.4
    output = cv2.addWeighted(img, alpha, output, 1 - alpha, 0, output)

    cv2.imwrite(out_file, output)
