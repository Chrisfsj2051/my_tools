_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/GLR_224x224_bs16.py',
    '../_base_/schedules/GLR_bs16_20k.py', '../_base_/default_runtime.py'
]

fp16 = dict(loss_scale='dynamic')

evaluation = dict(interval=1000, metric='accuracy')
data = dict(workers_per_gpu=4, samples_per_gpu=16)

# model settings
model = dict(
    head=dict(
        num_classes=203092
    )
)

evaluation = dict(interval=100000, metric='accuracy')
