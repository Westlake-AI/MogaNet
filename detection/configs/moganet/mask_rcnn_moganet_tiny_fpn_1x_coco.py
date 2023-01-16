_base_ = [
    '../_base_/models/mask_rcnn_moganet_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# optimizer
model = dict(
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
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
