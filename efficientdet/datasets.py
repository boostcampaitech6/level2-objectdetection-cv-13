import os
import json

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TrainDataset(Dataset):
    def __init__(
        self, 
        annot_path: str | os.PathLike, 
        root_dir: str, 
        transforms=None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        
        _coco_json = json.load(open(annot_path, 'r'))
        self.images = _coco_json['images']
        self.annotations = _coco_json['annotations']
        self.categories = _coco_json['categories']
        
    def __getitem__(self, index: int) -> None:
        img_info = self.images[index]
        img_id = img_info['id']
        
        img = cv2.imread(os.path.join(self.root_dir, img_info['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img /= 255.0
        
        ann_ids = list(filter(lambda x: x['image_id'] == img_id, self.annotations))
        boxes = np.array([x['bbox'] for x in ann_ids])
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes[:, [0, 2]] = boxes[:, [0, 2]]
        boxes[:, [1, 3]] = boxes[:, [1, 3]]
        
        labels = np.array([x['category_id'] for x in ann_ids])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        areas = np.array([x['area'] for x in ann_ids])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        is_crowds = np.array([x['iscrowd'] for x in ann_ids])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)
        
        target = {
            'img_id': torch.tensor([index]),
            'boxes': boxes, 
            'labels': labels,
            'area': areas,
            'iscrowd': is_crowds,
        }
        
        if self.transforms:
            while True:
                sample = self.transforms(**{
                    'image': img,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                })
                if len(sample['bboxes']) > 0:
                    img = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]
                    target['labels'] = torch.tensor(sample['labels'])
                    break
                
        return img, target, img_id
    
    def __len__(self) -> int:
        return len(self.images)
        

class InferenceDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''
    def __init__(
        self, 
        annot_path: str | os.PathLike, 
        root_dir: str, 
        transforms=None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        
        _coco_json = json.load(open(annot_path, 'r'))
        self.images = _coco_json['images']
        self.annotations = _coco_json['annotations']
        self.categories = _coco_json['categories']
        
    def __getitem__(self, index: int) -> None:
        img_info = self.images[index]
        img_id = img_info['id']
        
        img = cv2.imread(os.path.join(self.root_dir, img_info['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0

        # transform
        if self.transforms:
            sample = self.transforms(image=img)

        return sample['image'], img_id

    def __len__(self) -> int:
        return len(self.images)


class CustomTransformation:
    def __call__(self, method: str) -> A.Compose:
        if method == "train":
            return A.Compose([
                A.Resize(512, 512),
                A.Flip(p=0.5),
                ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        elif method == "valid":
            return A.Compose([
                A.Resize(512, 512),
                ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        else:
            return A.Compose([
                A.Resize(512, 512),
                ToTensorV2(p=1.0)
            ])
    