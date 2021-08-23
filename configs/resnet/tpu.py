_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/GLR_224x224_bs16.py',
    '../_base_/schedules/GLR_bs16_10e.py', '../_base_/default_runtime.py'
]

# fp16 = dict(loss_scale='dynamic')
# resume_from = 'work_dirs/debug/iter_520000.pth'
# evaluation = dict(interval=10000, metric='accuracy')
data = dict(workers_per_gpu=4, samples_per_gpu=32)

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet34')
    ),
    head=dict(
        num_classes=81313
    )
)

# total_iters = 2000*1000
# # optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='step', step=[total_iters//2, 3 * total_iters//4], by_epoch=False)
# runner = dict(type='IterBasedRunner', max_iters=total_iters)
