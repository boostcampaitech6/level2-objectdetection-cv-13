from typing import List, Union
from PIL import Image, ImageDraw
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

import copy
import wandb
import torch
import pytorch_lightning as pl
from transformers import DetaForObjectDetection, DetaImageProcessor
from coco_eval import CocoEvaluator

from utils import ID2LABEL, convert_to_xywh, prepare_for_coco_detection


# nn.Module
class DETA(pl.LightningModule):
    def __init__(
        self, 
        lr: float=0.001, 
        lr_backbone: float=0.0001, 
        weight_decay: float=0.001,
        processor=None,
        train_dataset=None,
        val_dataset=None
    ) -> None:   
        super().__init__()
        self.model = DetaForObjectDetection.from_pretrained(
            "jozhang97/deta-swin-large",
            num_labels=len(ID2LABEL)+1,
            # auxiliary_loss=True,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.processor: DetaImageProcessor = processor     

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        pixel_mask: torch.Tensor=None
    ) -> None:
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = None
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["gt_labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, outputs
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict, _ = self.common_step(batch, batch_idx)

        self.log("t_loss", loss, on_step=True, on_epoch=True)
        for k,v in loss_dict.items():
          self.log("t_" + k, v.item(), on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)

        if batch_idx % 30 == 0:
            # 이미지 및 bounding box를 WandB에 추가           
            self.log_wandb_images(batch, batch['gt_labels'], outputs)
            
        self.log("v_loss", loss, on_step=True, on_epoch=True)
        for k,v in loss_dict.items():
            self.log("v_" + k, v.item(), on_step=True, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.model.model.push_to_hub("zeroone012012/deta-swin-large")
        self.processor.push_to_hub("zeroone012012/deta-swin-large")
    
    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer
    
    def log_wandb_images(self, batch, labels, outputs):
        # Get relevant information from the batch and outputs
        pixel_values = batch["pixel_values"]
        logits = outputs["logits"]
        pred_boxes = outputs["pred_boxes"]

        # Convert logits to predicted labels
        predicted_labels = torch.argmax(logits, dim=2)

        # Loop through batch items
        for i in range(len(pixel_values)):
            normalized_pixel_values = pixel_values[i].permute(1, 2, 0).cpu().numpy()
            denormalized_pixel_values = ((normalized_pixel_values + 1) * 0.5 * 255).astype('uint8')
            image = Image.fromarray(denormalized_pixel_values).convert("RGB")
            draw = ImageDraw.Draw(image)

            # Draw ground truth bounding boxes
            # cx cy w h
            for box, label in zip(labels[i]["boxes"], labels[i]["class_labels"]):
                box = box*800.0
                cx, cy, w, h = box.cpu().numpy()
                x, y = cx-w/2, cy-h/2
                draw.rectangle([x, y, x + w, y + h], outline='green', width=2)
                draw.text((x, y), ID2LABEL[label.item()], fill='green')

            # Draw predicted bounding boxes
            for box, label in zip(pred_boxes[i], predicted_labels[i]):
                box = box*800.0
                cx, cy, w, h = box.cpu().numpy()
                x, y = cx-w/2, cy-h/2
                _label = ID2LABEL.get(label.item(), "background")
                if _label == "background":
                    continue
                draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
                draw.text((x, y), _label, fill='red')

            # Log the image to Wandb
            wandb.log({
                "image_with_boxes": wandb.Image(image, caption=f"Image with Ground Truth and Predicted Boxes (Batch {i})")
            })       
    
    def predict_step():
        pass


    