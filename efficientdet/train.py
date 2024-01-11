import os
import wandb
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import get_net, CustomOptimizer
from utils import Averager, collate_fn, get_config, set_seed, calculate_map
from datasets import TrainDataset, CustomTransformation
    

# train function
def train_fn(
    train_dataloader: DataLoader, 
    valid_dataloader: DataLoader,
    device: str, 
    config: dict,
    clip: int = 35
) -> None:
    model = get_net(config).to(device)
    optimizer = CustomOptimizer(config)(model, "Adam")   
    
    train_loss_hist = Averager()
    valid_loss_hist = Averager()
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss_hist.reset()
        valid_loss_hist.reset()
        # valid_preds, valid_targets = [], []
        
        for images, targets, _ in tqdm(train_dataloader):
            images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
            images = images.to(device).float()
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            target = {"bbox": boxes, "cls": labels}

            # calculate loss
            loss, cls_loss, box_loss = model(images, target).values()
            loss_value = loss.detach().item()
            train_loss_hist.send(loss_value)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            wandb.log({"train_cls_loss": cls_loss.detach().item()})
            wandb.log({"train_box_loss": box_loss.detach().item()})
        
        wandb.log({"train_loss": train_loss_hist.value})        
        
        model.eval()
        with torch.no_grad():
            for images, targets, _ in tqdm(valid_dataloader):
                images = torch.stack(images)
                images = images.to(device).float()
                boxes = [target['boxes'].to(device).float() for target in targets]
                labels = [target['labels'].to(device).float() for target in targets]
                target = {"bbox": boxes, "cls": labels, 'img_scale': None, 'img_size': None}

                loss, _, _, detections = model(images, target).values()
                loss_value = loss.detach().item()
                valid_loss_hist.send(loss_value)

                # valid_targets.extend(targets)
                # valid_preds.extend(detections)
                
        # mean_ap, c = calculate_map(valid_targets, valid_preds)
        wandb.log({"valid_loss": valid_loss_hist.value})
        # wandb.log({"valid_mAP_50": mean_ap})
        
        torch.save(
            model.state_dict(), 
            'ckpt/epoch_{}_tloss_{}_vloss_{}.pth'.format(
                epoch+1, train_loss_hist.value, valid_loss_hist.value
            )
        )
        
        
def main() -> None:
    ## load config
    config = get_config("cfg/default.yaml")
    set_seed(config["seed"])
    
    try:
        if os.path.exists(config['ckpt_dir']):
            raise Exception("ckpt_dir already exists")
    except:
        import sys
        print("ckpt_dir already exists")
        print("Remove ckpt_dir")
        os.rmdir(config['ckpt_dir'])
    else:
        os.mkdir(config["ckpt_dir"])
    
    # set wandb
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=config["wandb"]["name"],
        config=config,
    )
    
    # load dataset
    root_dir = config["root_dir"]
    train_file = config["train_file"]
    valid_file = config["valid_file"]
    
    train_dataset = TrainDataset(
        os.path.join(root_dir, train_file),
        root_dir,
        CustomTransformation()("train")
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    valid_dataset = TrainDataset(
        os.path.join(root_dir, valid_file),
        root_dir,
        CustomTransformation()("valid")
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Seed: {config['seed']}, Running on {device}")
    
    # train
    train_fn(train_dataloader, valid_dataloader, device, config)
    
    
if __name__ == '__main__':
    main()