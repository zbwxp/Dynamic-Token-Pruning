# model settings
_base_ = [
    '../_base_/datasets/ade20k_640x640.py',
    '../_base_/schedules/schedule_40k.py', 
    '../_base_/default_runtime.py'
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth' 
out_indices = [7, 15, 23]
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
img_size = 640
in_channels = 1024
thresh = 0.95
num_classes = 150
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[127.5, 127.5, 127.5], 
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,)
model = dict(
    type='EncoderDecoderPrune',
    pretrained=None,
    backbone=dict(
        type='ViT_prune',
        num_classes=num_classes,
        img_size=crop_size,
        patch_size=16,
        in_channels=3,
        embed_dims=in_channels,
        num_layers=24,
        num_heads=16,
        drop_path_rate=0.3,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        out_indices=out_indices,
        final_norm=False,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=False,
        interpolate_mode='bicubic',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix=None,
        )
    ),
    decode_head=dict(
        type='PruneHead',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels//2,
        num_classes=num_classes,
        thresh=thresh,
        num_heads=8,
        layers_per_decoder=3,
        loss_decode=dict(
            type='ATMLoss', num_classes=1, dec_layers=1, loss_weight=1.0),
    ),
    auxiliary_head=[
        dict(
            type='PruneHead',
            img_size=img_size,
            in_channels=in_channels,
            channels=in_channels//2,
            num_classes=num_classes,
            thresh=thresh,
            num_heads=8,
            layers_per_decoder=3,
            in_index=0,
            loss_decode=dict(
                type='ATMLoss', num_classes=1, dec_layers=1),
        ),
        dict(
            type='PruneHead',
            img_size=img_size,
            in_channels=in_channels,
            channels=in_channels//2,
            num_classes=num_classes,
            thresh=thresh,
            num_heads=8,
            layers_per_decoder=3,
            in_index=1,
            loss_decode=dict(
                type='ATMLoss', num_classes=1, dec_layers=1),
        ),
    ],
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(320, 320)),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR', 
        eta_min=0.0,
        power=0.9,
        begin=1500,
        end=40000,
        by_epoch=False)
]
log_processor = dict(
    by_epoch=False,
    window_size=50,
    custom_cfg=[
        dict(data_src='decode.acc_seg',
             method_name='mean',
            #  log_name='acc_seg_large_window',
             window_size=50)
    ],
)

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
find_unused_parameters=True