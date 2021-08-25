_base_ = [
    '../_base_/models/resnet152.py', '../_base_/datasets/GLR_224x224_bs16.py',
    '../_base_/schedules/GLR_bs16_10e.py', '../_base_/default_runtime.py'
]

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

optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=1.0))
data = dict(workers_per_gpu=1, samples_per_gpu=32)

