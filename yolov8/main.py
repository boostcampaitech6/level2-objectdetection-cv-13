import wandb
from glob import glob

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from argparse import ArgumentParser


def train():   
    wandb.init(
        project="Boost Camp Lv2-1",
        entity="frostings",
        name="yolov8m",
        notes="yolov8m with sample data augmentation",
    )
        
    model = YOLO("yolov8m.pt")
    model.train(
        data="cfg/default.yaml", epochs=10, imgsz=1280, 
        project="yolo_train", name="yolov8m", device=0,
        batch=16
    )
    

def inference():
    model = YOLO("yolo_train/yolov8m/weights/best.pt")
    infer_images = sorted(glob("../dataset/test/*.jpg"))
    prediction_strings = []
    file_names = []

    for idx, infer_image in tqdm(enumerate(infer_images)):
        img_id = '/'.join(infer_image.split('/')[1:])
        results = model.predict(infer_image, conf=0.05, iou=0.7)
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
    submission.to_csv("yolov8m_result.csv", index=False)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    
    args = parser.parse_args()
    if args.train:
        train()
    if args.inference:
        inference()