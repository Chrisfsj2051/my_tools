_base_ = [
    '../_base_/models/resnet152.py', '../_base_/datasets/GLR_224x224_bs16.py',
    '../_base_/schedules/GLR_bs16_20k.py', '../_base_/default_runtime.py'
]

# fp16 = dict(loss_scale='dynamic')
# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=10000)
evaluation = dict(interval=20000, metric='accuracy')
data = dict(workers_per_gpu=4, samples_per_gpu=64)

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet152')
    ),
    head=dict(
        num_classes=81313
    )
)
