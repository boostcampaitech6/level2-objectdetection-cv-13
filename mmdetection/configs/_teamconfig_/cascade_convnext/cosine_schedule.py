# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=0.00015, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise',
                                'num_layers': 12})
optimizer_config = dict(grad_clip=None) # in boostcamp default : dict(grad_clip=dict(max_keep_ckpts=3, interval=1))  
# learning policy
lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', 
    warmup_iters= 1000, 
    warmup_ratio= 1/10,
    min_lr=1e-07)