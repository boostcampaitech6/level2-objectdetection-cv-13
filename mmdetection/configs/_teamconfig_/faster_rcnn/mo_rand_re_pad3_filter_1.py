'''
 bbox_mAP: 0.0410, bbox_mAP_50: 0.0870, bbox_mAP_75: 0.0320, 
 bbox_mAP_s: 0.0000, bbox_mAP_m: 0.0040, 
 bbox_mAP_l: 0.0480, bbox_mAP_copypaste: 0.041 0.087 0.032 0.000 0.004 0.048
'''

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/level2-objectdetection-cv-13/dataset'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
img_scale = (512,512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),

    dict(type='RandomFlip', flip_ratio=0.5),

    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + '/k-fold-2024/train_fold0.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=classes,
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/k-fold-2024/valid_fold0.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
