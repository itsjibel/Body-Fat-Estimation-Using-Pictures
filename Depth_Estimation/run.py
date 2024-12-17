import argparse
import cv2
import numpy as np
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    print("Estimating pictures depth...", end=' ', flush=True)
    parser = argparse.ArgumentParser()
    INPUT_FOLDER = '../input'
    OUTPUT_FOLDER = 'output'

    args = parser.parse_args()

    margin_width = 50
    caption_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).to(DEVICE).eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    filenames = os.listdir(INPUT_FOLDER)
    filenames = [os.path.join(INPUT_FOLDER, filename) for filename in filenames if not filename.startswith('.')]
    filenames.sort()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        filename = os.path.basename(filename)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename[:filename.rfind('.')] + '_depth.png'), depth)

    print("Done!")