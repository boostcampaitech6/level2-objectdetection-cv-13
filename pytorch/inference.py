from pycocotools.coco import COCO

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

import os
import pandas as pd
from tqdm import tqdm

from dataset import CustomDataset


def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs


def main():
    annotation = '../../dataset/test.json' # annotation 경로
    data_dir = '../../dataset' # dataset 경로
    test_dataset = CustomDataset('test', annotation, data_dir)
    score_threshold = 0.05
    check_point = './checkpoints/faster_rcnn_torchvision_checkpoints.pth' # 체크포인트 경로
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # torchvision model 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 11  # 10 class + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(check_point))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    save_path = './result/faster_rcnn_torchvision_submission.csv'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    submission.to_csv(save_path, index=None)


if __name__ == '__main__':
    main()
