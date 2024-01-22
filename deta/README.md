## DETA Traning with Custom Dataset

### Installation
``` bash
python -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```python
python utils.py  ## prepare dataset
python python train.py  ## train
```

- Before run, you must login huggingface-cli and create repo.
- please change model git repo to your own huggingface model repo.
- Run inference.ipynb when inferencing your dataset
