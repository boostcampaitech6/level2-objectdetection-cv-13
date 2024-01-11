import os
import json
import random
import pandas as pd

import yaml
import torch
import numpy as np


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


def get_config(config_path: os.PathLike) -> dict:
    """Get config from config file.
    Args:
        config_path (os.PathLike): config file path
    Returns:
        config (dict): config dict
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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
    
    
def extract_sample_dataset(
    json_path: str | os.PathLike
) -> None:
    
    with open(json_path, "r") as f:
        coco_json = json.load(f)
    new_coco = {}
    images = coco_json["images"]
    annotations = coco_json["annotations"]
    categories = coco_json["categories"]
    
    new_coco["info"] = coco_json["info"]
    new_coco["licenses"] = coco_json["licenses"]
    new_coco["categories"] = categories
    new_coco["images"] = images[:100]
    new_annot = []
    for i in range(100):
        candits = list(filter(lambda x: x["image_id"] == i, annotations))
        new_annot.extend(candits)
    new_coco["annotations"] = new_annot
    
    with open("sample_data.json", "w") as f:
        json.dump(new_coco, f)


def calculate_map(
    targets: list, 
    predictions: list, 
) -> np.float64:
    """
    targets: list of dict
                dict: {'bbox': [num_bbox, 4], 'cls': [num_bbox]}
                y1 x1 y2 x2 => x, y, w, h
                
                [cls, ]
    
    predictions: [batch_size, max_bbox_per_image, 6]
                 6 => [label, score, x1, y1, x2, y2]
    """
    gt = []
    preds = []

    for t, prediction in zip(targets, predictions):
        img_id = t["img_id"].item()
        t["boxes"] = t["boxes"].cpu().numpy()
        t["labels"] = t["labels"].cpu().numpy()
        t["boxes"] *= 1024
        t["boxes"][:, 2] = t["boxes"][:, 2] + t["boxes"][:, 0]
        t["boxes"][:, 3] = t["boxes"][:, 3] + t["boxes"][:, 1]
        t["boxes"][:,[0,1,2,3]] = t["boxes"][:,[1,3,0,2]]
        gt.extend([[img_id, t["labels"][i], *t["boxes"][i]] for i in range(len(t["labels"]))])
        
        prediction = prediction.detach().cpu().numpy()
        prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
        for i in range(len(t["labels"])):
            predictions[i][2:] *= 1024
            preds.append([img_id, *prediction[i]])

    mean_ap, c = mean_average_precision_for_boxes(gt, preds, iou_threshold=0.5)
    return mean_ap, c
    
    
def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.5):
    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 6)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 7)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

    if isinstance(ann, str):
        valid = pd.read_csv(ann)
    else:
        valid = pd.DataFrame(ann, columns=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])

    if isinstance(pred, str):
        preds = pd.read_csv(pred)
    else:
        preds = pd.DataFrame(pred, columns=['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'])

    ann_unique = valid['ImageID'].unique()
    preds_unique = preds['ImageID'].unique()

    print('Number of files in annotations: {}'.format(len(ann_unique)))
    print('Number of files in predictions: {}'.format(len(preds_unique)))


    unique_classes = valid['LabelName'].unique().astype(np.str_)
    
    print('Unique classes: {}'.format(len(unique_classes)))

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)
    
    print('Detections length: {}'.format(len(all_detections)))
    print('Annotations length: {}'.format(len(all_annotations)))

    average_precisions = {}
    for _, label in enumerate(sorted(unique_classes)):

        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(ann_unique)):
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue

            num_annotations += len(annotations)
            detected_annotations = []

            annotations = np.array(annotations, dtype=np.float64)
            for d in detections:
                scores.append(d[4])

                if len(annotations) == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                overlaps = compute_overlap(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
            
    try:
        mean_ap = precision / present_classes
    except:
        mean_ap = 0.0
    print('mAP: {:.6f}'.format(mean_ap))
    
    return mean_ap, average_precisions


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_real_annotations(table):
    res = dict()
    ids = table['ImageID'].values.astype(np.str_)
    labels = table['LabelName'].values.astype(np.str_)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i]]
        res[id][label].append(box)

    return res


def get_detections(table):
    res = dict()
    ids = table['ImageID'].values.astype(np.str_)
    labels = table['LabelName'].values.astype(np.str_)
    scores = table['Conf'].values.astype(np.float32)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
        res[id][label].append(box)

    return res


def compute_overlap(boxes, query_boxes):
    """
    Args:
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps