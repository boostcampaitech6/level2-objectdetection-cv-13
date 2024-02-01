"""
root
├── dataset
│   ├── train
│   ├── test
│   ├── k-fold-2024
│   │   ├── train_fold0.csv
│   │   ├── valid_fold0.csv
│   │   ...
│   │
├── DETA
│   ├── coco
│   │   ├── train2017
│   │   ├── valid2017
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_valid2017.json
│   ├── prepare_dataset.py
│   ├── train.py
...
"""

import os
import json
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm
from transformers import DetaImageProcessor


def make_dataset(
    json_dir: str | os.PathLike,
    json_name: str,
    save_dir: str | os.PathLike,
    method: str
):
    """Change Dataset for training DETA.

    Args:
        json_dir (str | os.PathLike): kfold json dir
        json_name (str): kfold json name
        save_dir (str | os.PathLike): save image/annotation dir
        method (str): train / val
    Returns:
        None
    """
    
    # new dataset
    new_coco = {}
    new_images = []
    new_annotations = []
    new_categories = []
    
    # check dir
    if not os.path.exists(os.path.join(save_dir, method)):
        os.makedirs(os.path.join(save_dir, method))
    
    # load json file
    root_path = '/'.join(json_dir.split("/")[:-1])
    with open(os.path.join(json_dir, json_name), 'r') as f:
        json_data = json.load(f)
    
    # load data
    images = json_data["images"]
    annotations = json_data["annotations"]
    categories = json_data["categories"]
        
    # categories
    for category in categories:
        new_categories.append(category)
        
    annot_idx = 0
    # images and annotations
    for i, image in tqdm(enumerate(images)):
        # image
        new_img = {}
        new_img['id'] = i
        new_img['file_name'] = image['file_name'].split('/')[-1]
        new_img['height'] = image['height']
        new_img['width'] = image['width']
        new_images.append(new_img)
        
        # annotation
        candits = list(filter(lambda x: x['image_id'] == image['id'], annotations))
        for annot in candits:
            new_annot = {}
            new_annot['id'] = annot_idx
            new_annot['image_id'] = i
            new_annot['category_id'] = annot['category_id']
            new_annot['bbox'] = annot['bbox']
            new_annot['area'] = annot['area']
            new_annot['iscrowd'] = annot['iscrowd']
            new_annotations.append(new_annot)
          
            annot_idx += 1

        shutil.copy(
            os.path.join(root_path, image['file_name']),
            os.path.join(save_dir, method, new_img['file_name'])
        )
        
    new_coco['images'] = new_images
    new_coco['annotations'] = new_annotations
    new_coco['categories'] = new_categories
        
    with open(os.path.join(save_dir, f'{method}.json'), 'w') as f:
        json.dump(new_coco, f)
        

def extract_sample_dataset(
    json_path: str | os.PathLike
) -> None:
    
    new_coco = {}
    new_images = []
    new_annotations = []
    new_categories = []
    
    with open(json_path, "r") as f:
        coco_json = json.load(f)
        
    images = coco_json["images"]
    annotations = coco_json["annotations"]
    categories = coco_json["categories"]
    
    # categories
    for category in categories:
        new_categories.append(category)
    
    annot_idx = 0
    for i, image in tqdm(enumerate(images)):
        new_img = {}
        new_img['id'] = i
        new_img['file_name'] = image['file_name'].split('/')[-1]
        new_img['height'] = image['height']
        new_img['width'] = image['width']
        new_images.append(new_img)
        
        # annotation
        candits = list(filter(lambda x: x['image_id'] == image['id'], annotations))
        for annot in candits:
            new_annot = {}
            new_annot['id'] = annot_idx
            new_annot['image_id'] = i
            new_annot['category_id'] = annot['category_id']
            new_annot['bbox'] = annot['bbox']
            new_annot['area'] = annot['area']
            new_annot['iscrowd'] = annot['iscrowd']
            new_annotations.append(new_annot)
          
            annot_idx += 1
        
        if i == 100:
            break
        
    new_coco['images'] = new_images
    new_coco['annotations'] = new_annotations
    new_coco['categories'] = categories
        
    with open("coco/sample_data.json", "w") as f:
        json.dump(new_coco, f)


def convert_dataset(
    save_path: str | os.PathLike,
    json_path: str | os.PathLike,
    save_json: str | os.PathLike
) -> None:
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(json_path, "r") as f:
        json_data = json.load(f)
        
    images = json_data["images"]
    for idx, image in tqdm(enumerate(images)):
        shutil.copy(
            os.path.join("../dataset", image['file_name']),
            os.path.join("coco", image['file_name'])
        )
        image['file_name'] = image['file_name'].split('/')[-1]
        
    json_data["images"] = images
    with open(save_json, "w") as f:
        json.dump(json_data, f)

        
def collate_fn(batch, processor: DetaImageProcessor):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['gt_labels'] = labels
    return batch
        

def set_seed(seed: int):
    """Set seed for reproducibility.
    Args:
        seed (int): seed number
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


ID2LABEL = {
    0: "General trash", 1: "Paper", 2: "Paper pack", 
    3: "Metal", 4: "Glass", 5: "Plastic", 6: "Styrofoam", 
    7: "Plastic bag", 8: "Battery", 9: "Clothing"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}


if __name__ == '__main__':
    make_dataset(
        "../dataset/k-fold-2024",
        "valid_fold0.json",
        "coco",
        "val"
    )
    convert_dataset(
        "coco/train",
        "../dataset/new_train.json",
        "coco/train.json"
    )