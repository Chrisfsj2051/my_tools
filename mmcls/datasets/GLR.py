import copy
import warnings

from mmcls.core import support, precision_recall_f1
from mmcls.datasets import BaseDataset, DATASETS
import numpy as np
import pandas as pd
from mmcls.models import accuracy
import csv


@DATASETS.register_module()
class GoogleLandmarkDataset(BaseDataset):

    def __init__(self, **kwargs):
        super(GoogleLandmarkDataset, self).__init__(**kwargs)
        self.unique_ids = np.load('configs/unique_landmark_ids.np.npy')  # 81311
        self.id_map = {self.unique_ids[i]: i for i in range(len(self.unique_ids))}

    def process_filename(self, x):
        return f'{x[0]}/{x[1]}/{x[2]}/{x}.jpg'

    def load_annotations(self):
        ann_file = self.ann_file
        with open(ann_file, 'r') as f:
            content = f.readlines()
        content = [x.strip().split(' ') for x in content]
        data_infos = []
        for data in content:
            if len(data) == 1:
                data = (data[0], 1)
            data_infos.append(data)
            # filename, gt_label = data
            # gt_label = int(gt_label)
            # assert gt_label >= 1
            # # info = {'img_prefix': self.data_prefix}
            # info = {}
            # info['img_info'] = {'filename': filename}
            # # info['gt_label'] = np.array(gt_label - 1, dtype=np.int64)
            # info['gt_label'] = self.id_map[gt_label]
            # data_infos.append(info)
        return data_infos

    # def __len__(self):
    #     return 3000

    def prepare_data(self, idx):
        data = copy.deepcopy(self.data_infos[idx])
        filename = self.process_filename(data[0])
        gt_label = int(data[1])
        results = {}
        results['img_prefix'] = self.data_prefix
        results['gt_label'] = np.array(gt_label, dtype=np.int64)
        results['img_info'] = {'filename': filename}
        return self.pipeline(results)

    def format_results(self, results, thr):
        keys = [x['img_info']['filename'][6:-4] for x in self.data_infos]
        values = [(x[0][0], x[1][0]) for x in results]
        with open('submission.csv', 'w', newline='') as submission_csv:
            csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])
            csv_writer.writeheader()
            for image_id, prediction in zip(keys, values):
                label = self.unique_ids[prediction[0]]
                score = prediction[1]
                if score >= thr:
                    csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})
                else:
                    csv_writer.writerow({'id': image_id, 'landmarks': ''})

        # data_dict = dict(id=keys, landmarks=values)
        # data = pd.DataFrame(data_dict)
        # data.to_csv('submission.csv', index=False, quoting=csv.QUOTE_NONE, doublequote=False, escapechar=None, sep=',')

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results) // 2
        if len(gt_labels) != num_imgs:
            warnings.warn('Clip the length of test images')
            gt_labels = gt_labels[:num_imgs]

        # assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
        #                                    'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
