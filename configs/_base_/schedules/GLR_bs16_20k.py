total_iters = 200*1000
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[total_iters//2, 3 * total_iters//4], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=total_iters)
