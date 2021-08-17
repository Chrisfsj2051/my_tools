_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/GLR_224x224_bs16.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    head=dict(
        num_classes=203092
    )
)
