import os

import numpy as np
from PIL import Image, ImageDraw

import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DetaImageProcessor
from utils import collate_fn


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        annot_dir: str | os.PathLike, 
        annot_file: str | os.PathLike, 
        processor,
        transforms=None 
    ) -> None:
        ann_file = os.path.join("coco", annot_file)
        super(CocoDetection, self).__init__(annot_dir, ann_file)
        self.processor = processor
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    
## torch.utils.data.Dataset
class PLCocoDataset(pl.LightningDataModule):
    def __init__(
        self,
        train_transform=None,
        val_transform=None
    ) -> None:
        super().__init__()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.prepare_data_per_node = True
        
    def prepare_data(self) -> None:
        print("pass")
        pass
    
    def setup(self, stage=None):
        self.processor = DetaImageProcessor.from_pretrained("jozhang97/deta-swin-large")
        self.train_dataset = CocoDetection(
            annot_dir="coco/train",
            annot_file="train.json",
            processor=self.processor,
        )
        self.val_dataset = CocoDetection(
            annot_dir="coco/val",
            annot_file="val.json",
            processor=self.processor
        )
        return self.processor
  
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=lambda x: collate_fn(x, self.processor),
            batch_size=2,
            num_workers=8,
            shuffle=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=lambda x: collate_fn(x, self.processor),
            batch_size=2,
            num_workers=8,
            shuffle=False
        )


def vis_data(dataset: CocoDetection, method: str):
    img_ids = dataset.coco.getImgIds()
    img_id = img_ids[np.random.randint(0, len(img_ids))]
    img = train_dataset.coco.loadImgs(img_id)[0]
    print(img)
    img = Image.open(os.path.join("coco", method, img['file_name']))

    annots = dataset.coco.imgToAnns[img_id]
    draw = ImageDraw.Draw(img, "RGBA")
    cats = dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    
    for annot in annots:
        box = annot['bbox']
        cls_id = annot['category_id']
        x, y, w, h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        draw.text((x, y), id2label[cls_id], fill='white')
        
    img.save(f"{method}_test.jpg")
    
    
if __name__ == '__main__':
    processor = DetaImageProcessor.from_pretrained("facebook/detr-resnet-50")
    train_dataset = CocoDetection(
        annot_dir="coco/train",
        annot_file="train.json",
        processor=processor
    )
    
    val_dataset = CocoDetection(
        annot_dir="coco/val",
        annot_file="val.json",
        processor=processor
    )
    
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    
    # vis_data(train_dataset, "train")
    # vis_data(val_dataset, "val")
    