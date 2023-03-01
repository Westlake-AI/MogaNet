# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(        
        type='MogaNet_feat',
        arch="tiny",  # modify 'arch' for various architectures
        init_value=1e-5,
        drop_path_rate=0.1,
        stem_norm_cfg=norm_cfg,
        conv_norm_cfg=norm_cfg,
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 128, 256],  # modify 'in_channels'
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,  # modify 'in_channels' of stage-3
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
