# CV-13 Level 2 Object Detection 프로젝트

> ## 대회 개요: 지구 환경 보호를 위한 쓰레기 감지 모델 개발 대회 

### 배경
현대 사회는 대량 생산과 대량 소비의 시대로, 이는 '쓰레기 대란' 및 '매립지 부족'과 같은 다양한 환경 문제를 초래하고 있습니다. 이러한 문제에 대응하기 위해, 분리수거는 필수적인 활동으로 자리 잡았습니다. 정확하게 분류된 쓰레기는 자원으로 재활용될 수 있는 가치를 지니지만, 잘못 분류되면 매립지나 소각장으로 향하게 됩니다. 

### 목적
이 대회는 사진 속 쓰레기를 정확하게 감지하고 분류할 수 있는 모델을 개발하는 것을 목표로 합니다. 참가자들에게 제공되는 데이터셋은 다양한 종류의 쓰레기(일반 쓰레기, 플라스틱, 종이, 유리 등 10종)가 담긴 사진들로 구성되어 있으며, 이를 활용하여 모델을 학습시킬 수 있습니다. 

### 응용 가능성
성공적으로 개발된 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린이들의 분리수거 교육에 활용될 수 있습니다. 이를 통해 지구 환경 보호에 기여하고, 야생동물의 생계 위협과 같은 문제를 완화하는 데 도움을 줄 것입니다. 또한, 이 기술은 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술 등 다양한 분야에 응용될 수 있습니다. 

### 데이터셋과 평가 방법
- **Input**: 쓰레기 객체가 담긴 이미지 
- **Annotations**: bbox 정보 (좌표, 카테고리), COCO format으로 제공 
- **Output**: 모델은 bbox 좌표, 카테고리, score 값을 리턴해야 하며, 이를 바탕으로 submission 양식에 맞는 csv 파일을 생성하여 제출 

### 요약
이 대회는 지구 환경 보호에 기여할 수 있는 쓰레기 감지 및 분류 모델을 개발하는 것을 목표로 합니다. 제공되는 데이터셋을 활용해 모델을 학습시키고, 이를 통해 분리수거를 효율적으로 할 수 있는 방법을 탐색합니다. 이 기술은 쓰레기 처리 시설의 효율성을 높이고, 환경 교육에도 응용될 수 있습니다. 참가자들은 지구 환경 보호를 위한 중요한 일에 기여하는 동시에, 인공지능과 기계 학습 분야에서 실질적인 문제 해결 능력을 키울 수 있는 기회를 갖게 됩니다. 

> ## Project Structure

```
📦level2-objectdetection-cv-13
┣ 📂 eda
┃ ┣ 📜 eda.ipynb
┃ ┣ 📜 labelme_to_coco.py
┃ ┣ 📜 object_add_algorithm.ipynb
┃ ┣ 📜 show_image.ipynb
┃ ┣ 📜 show_test_image.ipynb
┣ 📂 pytorch
┣ 📂 detectron2
┣ 📂 deta
┣ 📂 efficientdet
┣ 📂 yolov8
┣ 📂 mmdetection
┃ ┣ 📂configs
┃ ┃ ┣ 📂_base_
┃ ┃ ┣ 📂_teamconfig_
┃ ┣ 📂mmdet
┃ ┗ 📜train.py
┃ ┗ 📜test.py
┣ 📜 small_box.ipynb
┣ 📜 upsampling.ipynb
┣ 📜 wbf_ensemble.ipynb
┣ 📜 Train-valid_Split.ipynb
┗ 📜 README.md
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
