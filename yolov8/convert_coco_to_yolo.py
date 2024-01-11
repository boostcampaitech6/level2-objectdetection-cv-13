import os
import json
import shutil

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold


def split_straitify(root_dir: str, json_file: str):
    with open(os.path.join(root_dir, json_file), 'r') as f:
        coco_json = json.load(f)
        
    coco_json["annotations"] = [annot for annot in coco_json["annotations"] if annot["area"] < 400000 and annot["area"] > 550]
    for idx in range(len(coco_json["annotations"])):
        coco_json["annotations"][idx]["id"] = idx
    
    var = [(ann['image_id'], ann['category_id']) for ann in coco_json["annotations"]]
    X = np.ones((len(coco_json["annotations"]),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])
    annots = coco_json["annotations"]    
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2024)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_idx, val_idx = set(train_idx), set(val_idx)
        train_img_ids = set([ann['image_id'] for ann in list(filter(lambda x: x["id"] in train_idx, annots))])
        val_img_ids = set([ann['image_id'] for ann in list(filter(lambda x: x["id"] in val_idx, annots))])
        break
    
    return train_img_ids, val_img_ids


def convert_coco_to_yolo_format(root_dir: str, json_file: str, save_dir: str, img_ids: list = None):
    # Check directory
    try:
        assert os.path.exists(os.path.join(root_dir, save_dir, "images")) == True
        assert os.path.exists(os.path.join(root_dir, save_dir, "labels")) == True
    except:
        os.makedirs(os.path.join(root_dir, save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, save_dir, "labels"), exist_ok=True)
    finally:
        print("Finish make directory")
    
    # Load json
    with open(os.path.join(root_dir, json_file), 'r') as f:
        coco_json = json.load(f)
    anoots = coco_json["annotations"]
    
    print("Start converting...")
    for image in tqdm(sorted(coco_json["images"], key=lambda x: x["id"])):
        w, h, file_name, image_id = image["width"], image["height"], image["file_name"], image["id"]
        if img_ids is not None and image_id not in img_ids:
            continue
        file_name = file_name.split("/")[1]

        # filtering annotations 
        obj_candits = list(filter(lambda x: x["image_id"] == image_id, anoots))
        
        # Save txt format to train yolo
        with open(os.path.join(root_dir, save_dir, "labels", f"{file_name[:-4]}.txt"), "w") as f:
            for obj_candit in obj_candits:
                # x1 y1 w h -> cx cy w h
                cat_id = obj_candit["category_id"]
                x1, y1, width, height = obj_candit["bbox"]
                scaled_cx, scaled_cy = (x1+width/2) / w, (y1+height/2) / h
                scaled_width, scaled_height = width / w, height / h                
                f.write("%s %.3f %.3f %.3f %.3f\n" %(cat_id, scaled_cx, scaled_cy, scaled_width, scaled_height))
                
            f.close()

        # Copy image to new directory
        shutil.copy(os.path.join(root_dir, "train", file_name), os.path.join(root_dir, save_dir, "images", file_name))
    print("Finish converting...")
        

if __name__ == '__main__':
    # train_img_ids, val_img_ids = split_straitify("../dataset", "new_train.json")
    # convert_coco_to_yolo_format("../dataset", "new_train.json", "yolo_train", sorted(train_img_ids))
    # convert_coco_to_yolo_format("../dataset", "new_train.json", "yolo_eval", sorted(val_img_ids))
    convert_coco_to_yolo_format("../dataset", "k-fold-2024/train_fold0.json", "yolo_train", None)
    convert_coco_to_yolo_format("../dataset", "k-fold-2024/valid_fold0.json", "yolo_eval", None)