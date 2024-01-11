import os
import wandb
from glob import glob

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from argparse import ArgumentParser


def train(config: list[str | os.PathLike, str]):   
    wandb.init(
        project="Boost Camp Lv2-1",
        entity="frostings",
        name=f"yolov8x_{config[1]}",
        notes="yolov8x with sample data augmentation",
    )
        
    model = YOLO("yolov8x.pt")
    model.train(
        data=config[0], epochs=50, imgsz=640, 
        project="yolo_train", name=f"yolov8x_{config[1]}", device=0,
        batch=32
    )
    

def inference(config: list[str | os.PathLike, str]): 
    model = YOLO(f"yolo_train/yolov8x_{config[1]}/weights/best.pt")
    infer_images = sorted(glob("../dataset/test/*.jpg"))
    prediction_strings = []
    file_names = []

    for idx, infer_image in tqdm(enumerate(infer_images)):
        img_id = '/'.join(infer_image.split('/')[2:])
        results = model.predict(infer_image, conf=0.05)
        boxes = results[0].boxes
        
        prediction_string = ''
        for box in boxes:
            class_id = box.cls.cpu().numpy()[0]
            score = box.conf.cpu().numpy()[0]
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            prediction_string += str(int(class_id)) + ' ' + str(score) + ' ' + str(x1) + ' ' + str(
                        y1) + ' ' + str(x2) + ' ' + str(y2) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(img_id)

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(f"yolov8x_{config[1]}_result.csv", index=False)
    
    
if __name__ == '__main__':
    configs = [
        ["cfg/fold0.yaml", "fold0"],
        ["cfg/fold1.yaml", "fold1"],
        ["cfg/fold2.yaml", "fold2"],
        ["cfg/fold3.yaml", "fold3"],
        ["cfg/fold4.yaml", "fold4"]
    ]
    
    for i, config in enumerate(configs):
        train(config)
        inference(config)
    