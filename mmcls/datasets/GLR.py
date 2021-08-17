from mmcls.datasets import BaseDataset, DATASETS
import numpy as np


@DATASETS.register_module()
class GoogleLandmarkDataset(BaseDataset):

    def load_annotations(self):
        def process_filename(x):
            return f'{x[0]}/{x[1]}/{x[2]}/{x}'

        ann_file = self.ann_file
        with open(ann_file, 'r') as f:
            content = f.readlines()
        content = [x.strip().split(' ') for x in content]
        data_infos = [{process_filename(x[0]): np.array(x[1], dtype=np.int64)} for x in content]
        return data_infos

