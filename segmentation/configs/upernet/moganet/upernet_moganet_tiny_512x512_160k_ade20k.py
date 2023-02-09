_base_ = [
    '../../_base_/models/upernet_moganet.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MogaNet_feat',
        arch='tiny',
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=\
                'https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz224_8xbs128_ep300.pth.tar',
            ),
        ),
    decode_head=dict(
        in_channels=[32, 64, 128, 256],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=128,
        num_classes=150,
    ))

# AdamW optimizer, no weight decay for position embedding & norm & layer scale in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'layer_scale': dict(decay_mult=0.),
                                                 'scale': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU for bs16
data = dict(samples_per_gpu=2)
evaluation = dict(save_best='auto')
