# CV-13 Level 2 Object Detection 프로젝트

> ## Project Structure

```
📦level2-objectdetection-cv-13
 ┣ 📂eda
 ┃ ┣ 📜eda.ipynb
 ┃ ┗ 📜show_image.ipynb
 ┣ 📂pytorch
 ┃ ┣ 📜faster_rcnn_torchvision_inference.ipynb
 ┃ ┗ 📜faster_rcnn_torchvision_train.ipynb
 ┣ 📂detectron2
 ┣ 📂mmdetection
 ┃ ┣ 📂configs
 ┃ ┣ 📂mmdet
 ┃ ┗ 📜train.py
 ┗ 📜README.md
```

> ## DINO 실행 방법 및 코드

### 1. `dino_requirements.txt` 설치

```
pip install -r dino_requirements.txt
```

### 2. `Train.py` 실행

```
python mmdetectionv3/train.py {config 폴더 경로} --work-dir {모델이 저장될 경로}

# DINO Train:

python mmdetectionv3/train.py mmdetectionv3/configs/_teamconfig_/dinonoval/config.py --work-dir results/dino
```

### 3. `Test.py` 실행

```
python mmdetectionv3/test.py {config 폴더 경로} {모델 체크포인트 경로} --work-dir {결과가 저장될 경로}

# DINO Test:

python mmdetectionv3/test.py mmdetectionv3/configs/_teamconfig_/dinonoval/config.py results/dino/best.pth --work-dir out/dino
```


> ## Cascade R-CNN 실행 방법 및 코드

### 1. `cascade_requirements.txt` 설치

```
pip install -r cascade_requirements.txt
```

### 2. `Train.py` 실행

```
python mmdetection/train.py --config {_teamconfig_안에 있는 폴더명}/{config파일명} --exp-name {실험명} --work-dir {모델을 저장할 경로} --epochs {epoch 갯수}

# Cascade Train:

python mmdetection/train.py --config cascade_rcnn/cascade_rcnn_config.py --exp-name cascade --work-dir results/cascade --epochs 15
```

### 3. `Test.py` 실행

```
python mmdetection/test.py --config {_teamconfig_안에 있는 폴더명}/{config파일명} --checkpoint {모델이 저장된 경로}

# Cascade Test:

python mmdetection/test.py --config cascade_rcnn/cascade_rcnn_config.py --checkpoint results/cascade/best.pth
```


> ## Ensemble 방법

- wbf_ensemble.ipynb 파일에서 csv파일 경로를 지정한 후 앙상블하실 수 있습니다.

---

> ## 최종 모델
<p align = "center">
<img height="400px" width="800px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-13/assets/78347296/9ed84f59-178b-4fad-844a-684a7b47557c">
<p/>

### K-fold Ensemble of Cascade R-CNN with Swin-Large Backbone + DINO with Swin-Large Backbone

- Swin-Large를 Backbone으로 사용한 Cascade R-CNN은 5개의 폴드로 나누어 WBF Ensemble을 진행하였습니다.

- Swin-Large를 Backbone으로 사용한 DINO는 Train set 전체를 사용하였습니다.

- 최종적으로 위 두 모델을 NMS Ensemble하였고, 그 결과 리더보드 Public mAP_50 0.7013, Private mAP_50 0.6850을 기록하였습니다.
