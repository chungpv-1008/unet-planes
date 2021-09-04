import logging
import cv2
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import parse_xml_image, label_in_box


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, xmls_dir: str, scale: float = 1.0, mask_suffix: str = '', xml_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.xmls_dir = Path(xmls_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.xml_suffix = xml_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def load_mask(cls, mask_path):
        return cv2.imread(mask_path)[..., ::-1]

    @classmethod
    def load_image(cls, image_path):
        return Image.open(image_path)

    @classmethod
    def preprocess_mask(cls, mask_img, scale, xml_path, mask_path):
        df = parse_xml_image(xml_path)
        # get the bboxes
        bboxes = np.array([df['xmin'], df['xmax'], df['ymin'], df['ymax']]).transpose()
        # get the segmentation
        colors = None
        colors = df['object_mask_color_rgba']

        height, width, _ = mask_img.shape
        new_height, new_width = int(height * scale), int(width * scale)
        assert new_height > 0 and new_width > 0, 'Scale is too small, resized images would have no pixel'
        mask_img = cv2.resize(mask_img, (new_width, new_height))

        mask = np.zeros(shape=(new_height, new_width))
        # create the annotations
        for idx in range(bboxes.shape[0]):
            color = colors[idx]
            mask += label_in_box(color, mask_img)
        mask = np.where(mask < 1., 0., 1.)
        return mask

    @classmethod
    def preprocess_image(cls, image, scale):
        width, height = image.size
        new_width, new_height = int(scale * width), int(scale * height)
        assert new_width > 0 and new_height > 0, 'Scale is too small, resized images would have no pixel'

        image = image.resize((new_width, new_height))
        image_array = np.asarray(image)
        if image_array.ndim == 2:
            image_array = image_array[np.newaxis, ...]
        elif image_array.shape[-1] == 3:
            image_array = image_array.transpose((2, 0, 1))
        elif image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3].transpose((2, 0, 1))
        return image_array / 255

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        xml_file = list(self.xmls_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(xml_file) == 1, f'Either no mask or multiple xmls found for the ID {name}: {xml_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        mask_image = self.load_mask(str(mask_file[0]))
        image = self.load_image(img_file[0])

        mask = self.preprocess_mask(mask_image, self.scale, xml_file[0], mask_file[0])
        image = self.preprocess_image(image, self.scale)

        assert image.shape[1] == mask.shape[0] and image.shape[2] == mask.shape[1], \
            'Image and mask {name} should be the same size, but are ({image.shape[1]}, {image.shape[2]}) and ({mask.shape[0]}, {mask.shape[1]})'

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, xmls_dir, scale=1):
        super().__init__(images_dir, masks_dir, xmls_dir, scale, mask_suffix='_mask', xml_suffix='')
