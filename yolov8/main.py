import os
import sys
from glob import glob
from argparse import ArgumentParser

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion


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
    
    
def wbf_ensemble():
    submission_files = glob("*.csv")
    submission_df = [pd.read_csv(file) for file in submission_files]
    
    try:
        image_ids = submission_df[0]['image_id'].tolist()
        assert len(image_ids)==4871
    except:
        print("Submission csv length Error")
        sys.exit(0)
    
    annotation = '../dataset/test.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []

    for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        prediction_string = ''
        boxes_list, scores_list, labels_list = [], [], []
        image_info = coco.loadImgs(i)[0]
        
        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue
                
            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info['width']
                box[1] = float(box[1]) / image_info['height']
                box[2] = float(box[2]) / image_info['width']
                box[3] = float(box[3]) / image_info['height']
                box_list.append(box)
                
            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))
        
        if len(boxes_list):
            # Ensemble Boxes는 아래와 같은 메서드도 지원합니다! (기본 nms, soft-nms 등등)
            
            # boxes, scores, labels = nms(boxes_list, scores_list, labels_list,iou_thr=iou_thr)
            # boxes, scores, labels = soft_nms(box_list, scores_list, labels_list, iou_thr=iou_thr)
            # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list,iou_thr=iou_thr)
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,iou_thr=0.55)

            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
        
        prediction_strings.append(prediction_string)
        file_names.append(image_id)
    
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    #======= 앙상블 결과 출력할 경로 지정 ========#
    submission.to_csv('yolov8x_kfold_wbf_ensemble.csv', index=False)
    #===========================================#
    submission.head()

    
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
    
    wbf_ensemble()