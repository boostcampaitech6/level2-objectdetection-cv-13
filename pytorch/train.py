import os

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
import code
# code.interact(local=dict(globals(), **locals()))

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


def train_fn(num_epochs, train_data_loader, optimizer, model, device):
    best_loss = float('inf')
    loss_hist = Averager()
    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = './models/best.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value
    

def main():
    annotation = '../../dataset/train.json' # annotation path
    data_dir = '../../dataset' # data_dir path
    train_dataset = CustomDataset(annotation, data_dir, train=True) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load torchvision model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 11 # class num = 10 + background
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    num_epochs = 1
    
    # training
    train_fn(num_epochs, train_data_loader, optimizer, model, device)


if __name__ == '__main__':
    main()
