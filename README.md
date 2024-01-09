# CV-13 Level 2 Object Detection 프로젝트 베이스라인 코드

## Project Structure

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
## Train

`python mmdetection/train.py --config {_teamconfig_안에 있는 폴더명}/{config파일명} --work-dir {모델을 저장할 경로} --epochs {epoch 갯수}`

## Test(Inference)

`python mmdetection/test.py --config {_teamconfig_안에 있는 폴더명}/{config파일명} --checkpoint {모델이 저장된 경로}`
