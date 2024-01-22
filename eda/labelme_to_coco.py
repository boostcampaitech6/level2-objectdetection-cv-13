import os
import json
from glob import glob
from tqdm import tqdm


root_dir = "__path_to_your_labelme_json_files__"
json_list = glob(os.path.join(root_dir, "*.json"))
json_list = sorted(json_list)

CATEGORIES = {
    "General trash": 0, 
    "Paper": 1,
    "Paper pack": 2,
    "Metal": 3,
    "Glass": 4,
    "Plastic": 5,
    "Styrofoam": 6,
    "Plastic bag": 7,
    "Battery": 8,
    "Clothing": 9,
}

images = []
annotations = []
categories = [
    {'id': 0, 'name': 'General trash', 'supercategory': 'General trash'},
    {'id': 1, 'name': 'Paper', 'supercategory': 'Paper'},
    {'id': 2, 'name': 'Paper pack', 'supercategory': 'Paper pack'},
    {'id': 3, 'name': 'Metal', 'supercategory': 'Metal'},
    {'id': 4, 'name': 'Glass', 'supercategory': 'Glass'},
    {'id': 5, 'name': 'Plastic', 'supercategory': 'Plastic'},
    {'id': 6, 'name': 'Styrofoam', 'supercategory': 'Styrofoam'},
    {'id': 7, 'name': 'Plastic bag', 'supercategory': 'Plastic bag'},
    {'id': 8, 'name': 'Battery', 'supercategory': 'Battery'},
    {'id': 9, 'name': 'Clothing', 'supercategory': 'Clothing'}
]

annot_idx = 0
for idx, json_file in tqdm(enumerate(json_list)):
    with open(json_file, "r") as f:
        data = json.load(f)
        
    image = {}
    image["id"] = idx
    image['width'] = data["imageWidth"]
    image['height'] = data["imageHeight"]
    image['file_name'] = "train/"+data["imagePath"]
    images.append(image)
    
    for annot in data['shapes']:
        annotation = {}
        annotation['image_id'] = idx
        annotation['category_id'] = CATEGORIES[annot['label']]
        
        x1, y1, x2, y2 = *annot['points'][0], *annot['points'][1]
        x1, y1, x2, y2 = round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)
        w, h = round(x2 - x1, 1), round(y2 - y1, 1)
        area = round(w * h, 2)
        annotation['area'] = area
        annotation['bbox'] = [x1, y1, w, h]
        annotation['iscrowd'] = 0
        annotation['id'] = annot_idx
        
        annotations.append(annotation)
        annot_idx += 1

new_coco = {}
new_coco["images"] = images
new_coco["annotations"] = annotations
new_coco["categories"] = categories

with open("new_train.json", "w") as f:
    json.dump(new_coco, f)    
    