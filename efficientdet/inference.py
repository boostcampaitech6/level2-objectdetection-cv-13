import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import load_net
from utils import collate_fn, get_config, set_seed
from datasets import CustomTransformation, InferenceDataset


def inference(
    test_dataloader: DataLoader, 
    ckpt: str | os.PathLike, 
    device: str, 
    config: dict
) -> None:
    model = load_net(
        config, 
        ckpt
    ).to(device)
    
    outputs = []  
    model.eval()
    for images, img_ids in tqdm(test_dataloader):
        images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({
                "boxes": out.detach().cpu().numpy()[:, :4],
                "scores": out.detach().cpu().numpy()[:, 4],
                "labels": out.detach().cpu().numpy()[:, 5]
            })
            
    print(outputs)
            
    prediction_strings = []
    file_names = []
    with open(os.path.join(config["root_dir"], config["valid_file"]), 'r') as f:
        annotations = json.load(f)

    for i, output in enumerate(outputs):
        prediction_string = ""
        image_info = annotations["images"][i]
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
            if score > 0.05:
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'efficientdet.csv', index=None)
    print(submission.head())
        
        
if __name__ == '__main__':
    config = get_config("cfg/default.yaml")
    set_seed(config["seed"])
    
    # load dataset
    root_dir = config["root_dir"]
    valid_file = config["valid_file"]
    ckpt_dir = config["ckpt_dir"]
    ckpt_file = config["ckpt_file"]
    
    test_dataset = InferenceDataset(
        annot_path=os.path.join(root_dir, valid_file),
        root_dir=root_dir,
        transforms=CustomTransformation()("test")
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed: {config['seed']}, Running on {device}")
    
    inference(
        test_dataloader,
        os.path.join(ckpt_dir, ckpt_file), # ckpt/epoch_1_tloss0.0_vloss0.0.pth
        device,
        config
    )
    