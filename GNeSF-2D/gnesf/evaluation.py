import numpy as np

from detectron2.evaluation import SemSegEvaluator


class SemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        super().__init__(dataset_name, distributed=distributed, output_dir=output_dir)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int64)
            gt = np.array(input['sem_seg'], dtype=np.int64)

            gt[gt == self._ignore_label] = self._num_classes
            # print(pred)

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


    def process_nerf(self, input, output):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        output = output["sem"].argmax(dim=-1).to(self._cpu_device)
        pred = np.array(output, dtype=np.int64)
        
        gt = np.array(input['sem'][0], dtype=np.int64)

        gt[gt == self._ignore_label] = self._num_classes

        self._conf_matrix += np.bincount(
            (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=self._conf_matrix.size,
        ).reshape(self._conf_matrix.shape)

        if self._compute_boundary_iou:
            b_gt = self._mask_to_boundary(gt.astype(np.uint8))
            b_pred = self._mask_to_boundary(pred.astype(np.uint8))

            self._b_conf_matrix += np.bincount(
                (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

        self._predictions.extend(self.encode_json_sem_seg(pred, input["rgb_path"]))