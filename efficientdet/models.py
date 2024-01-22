import torch
from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from lion_pytorch  import Lion


# Effdet config
# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py
def get_net(cfg: dict, checkpoint_path=None) -> DetBenchTrain:
    config = get_efficientdet_config(cfg['model_name'])
    config.num_classes = cfg['num_classes']
    config.image_size = (cfg['input_size'], cfg['input_size'])

    config.soft_nms = cfg['soft_nms']
    config.max_det_per_image = cfg['max_det_per_image']

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])

    return DetBenchTrain(net)


def load_net(cfg: dict, checkpoint_path):
    config = get_efficientdet_config(cfg['model_name'])
    config.num_classes = cfg['num_classes']
    config.image_size = (cfg['input_size'], cfg['input_size'])

    config.soft_nms = cfg['soft_nms']
    config.max_det_per_image = cfg['max_det_per_image']
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net


class CustomOptimizer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        
    def __call__(self, model: DetBenchTrain, method: str="SGD"):
        _cfg = self.cfg["optimizer"][method]
        if method == "SGD":
            return torch.optim.SGD(
                model.parameters(), 
                lr=_cfg['lr'], 
                momentum=_cfg['momentum'], 
                weight_decay=_cfg['weight_decay']
            )
        elif method == "Adam":
            return torch.optim.Adam(
                model.parameters(), 
                lr=_cfg['lr'], 
                betas=(_cfg['beta1'], _cfg['beta2']), 
                eps=_cfg['eps'], 
                weight_decay=_cfg['weight_decay']
            )
        elif method == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), 
                lr=_cfg['lr'], 
                betas=(_cfg['beta1'], _cfg['beta2']), 
                eps=_cfg['eps'], 
                weight_decay=_cfg['weight_decay']
            )
        elif method == "Lion":
            return Lion(
                model.parameters(), 
                lr=_cfg['lr'], 
                betas=(_cfg['beta1'], _cfg['beta2']),
                weight_decay=_cfg['weight_decay']
            )