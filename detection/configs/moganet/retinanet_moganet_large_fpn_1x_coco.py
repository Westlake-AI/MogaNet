_base_ = [
    '../_base_/models/retinanet_moganet_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        type='MogaNet_feat',
        arch='large',
        drop_path_rate=0.3,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=\
                'https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_large_sz224_8xbs64_ep300.pth.tar', 
            ),
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 160, 320, 640],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
