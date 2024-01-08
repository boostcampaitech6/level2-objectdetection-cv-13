## YOLOv8 Traning with Custom Dataset

### Installation
``` bash
python -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```


### Usage

```python
python convert_coco_to_yolo.py
python main.py --train  # train
python main.py --inference # inference
```

### Caution

- change datasets_dir in ".config/Ultralytics/setting.yaml" to "~~~/level2-objectdetection-cv-13" if error like "Dataset 'cfg/default.yaml' images not found"