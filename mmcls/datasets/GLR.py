from mmcls.datasets import BaseDataset, DATASETS
import numpy as np


@DATASETS.register_module()
class GoogleLandmarkDataset(BaseDataset):

    def load_annotations(self):
        def process_filename(x):
            return f'{x[0]}/{x[1]}/{x[2]}/{x}.jpg'

        ann_file = self.ann_file
        with open(ann_file, 'r') as f:
            content = f.readlines()
        content = [x.strip().split(' ') for x in content]
        data_infos = []
        for filename, gt_label in content:
            gt_label = int(gt_label)
            assert gt_label >= 1
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': process_filename(filename)}
            info['gt_label'] = np.array(gt_label - 1, dtype=np.int64)
            data_infos.append(info)
        return data_infos
