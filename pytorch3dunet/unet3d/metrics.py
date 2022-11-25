import importlib

import numpy as np
import torch
from skimage import measure
from skimage.metrics import adapted_rand_error, peak_signal_noise_ratio, mean_squared_error

from pytorch3dunet.unet3d.losses import compute_per_channel_dice
from pytorch3dunet.unet3d.seg_metrics import AveragePrecision, Accuracy
from pytorch3dunet.unet3d.utils import get_logger, expand_as_one_hot, convert_to_numpy

logger = get_logger('EvalMetric')


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class AdaptedRandError:
    """
    A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
    (http://brainiac2.mit.edu/SNEMI3D/evaluation).

    This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_seg()`)
    and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
    it's enough to extend this class and implement the `input_to_seg` method.

    NOTE: The input is converted into Numpy Array first and back to Torch Tensor when returned.
    """

    def __init__(self, input_channels, target_channels, ignore_index=None, **kwargs):
        if input_channels is None or target_channels is None:
            raise ValueError("AdaptedRandError metric: input_/target_channels cannot be None.")
        if input_channels != target_channels:
            raise ValueError("AdaptedRandError metric, input_channels != target_channels.")

        self.input_channels = input_channels
        self.target_channels = target_channels
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        Compute ARand Error for each input, target pair in the batch and return the mean value.

        Args:
            input (torch.tensor): 5D (NCDHW) output from the network
            target (torch.tensor): 5D (NCDHW) ground truth segmentation

        Returns:
            average ARand Error across the batch
        """

        def _arand_err(gt, seg):
            return adapted_rand_error(gt, seg)[0]

        # convert numpy arrays and select specified channels
        input, target = convert_to_numpy(input, target)

        if input.ndim != 5 or target.ndim != 5:
            raise ValueError("AdaptedRandError only supports 5D (NCDHW) input and target.")

        input = input[:, self.input_channels, ...]  # dim not changed
        target = target[:, self.target_channels, ...]  # dim not changed

        # ensure target is of integer type
        target = target.astype(np.int)
        if self.ignore_index is not None:
            target[target == self.ignore_index] = 0

        # For each batch of N samples, get average error
        per_batch_arand = []
        for input_i, target_i in zip(input, target):  # For CDHW in NCDHW

            # For each sample of C channels, get sum of channel-wise errors
            per_channel_arand = []
            for input_ic, target_ic in zip(input_i, target_i):  # For DHW in CDHW

                # skip ARand for single-valued target to avoid zero-division
                if np.all(target_ic == target_ic[0]):
                    logger.info('Skipping ARandError computation: only 1 label present in the ground truth')
                    per_batch_arand.append(0.0)
                    continue

                # convert input_ic to segmentation TDHW, where T is for 'Threshold'
                segs = self.input_to_seg(input_ic)
                assert segs.ndim == 4

                # compute per threshold arand 
                per_threshold_arand = [_arand_err(target_ic, seg_c) for seg_c in segs]

                # add the minimum arand, among all threshold, for this channel
                per_channel_arand.append(np.min(per_threshold_arand))

            # add the sum of arand across channels as the arand for this sample
            per_batch_arand.append(np.sum(per_channel_arand))

        # return mean arand error across all samples in this batch
        mean_arand = torch.mean(torch.tensor(per_batch_arand))
        logger.info(f'ARand: {mean_arand.item()}')
        return mean_arand

    def input_to_seg(self, input):
        """
        Converts input tensor (output from the network) to the segmentation image. E.g. if the input is the boundary
        pmaps then one option would be to threshold it and run connected components in order to return the segmentation.

        :param input: 4D tensor (CDHW)
        :return: segmentation volume either 4D (segmentation per channel)
        """
        # by deafult assume that input is a segmentation volume itself
        return input


class BoundaryAdaptedRandError(AdaptedRandError):
    """
    Compute ARand between the input boundary map and target segmentation.
    Boundary map is thresholded, and connected components is run to get the predicted segmentation
    """

    def __init__(
        self,
        input_channels=None,
        target_channels=None,
        ignore_index=None,
        thresholds=None,
        invert_pmaps=False,
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            target_channels=target_channels,
            ignore_index=ignore_index,
            **kwargs,
        )

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.invert_pmaps = invert_pmaps

    def input3d_to_instance_segmentation(self, prediction):
        """
        Input a 3D (ZYX) volume of boundary semantic prediction
        Return a 4D (TZYX) volume of boundary instance segmentation
        """
        logger.info("input3d_to_instance_segmentation() is called for metric.")
        segs = []
        for th in self.thresholds:
            # threshold probability maps
            prediction = prediction > th

            if self.invert_pmaps:
                # for connected component analysis we need to treat boundary signal as background
                # assign 0-label to boundary mask
                prediction = np.logical_not(prediction)

            prediction = prediction.astype(np.uint8)
            # run connected components on the predicted mask; consider only 1-connectivity
            seg = measure.label(prediction, background=0, connectivity=1)
            segs.append(seg)

        return np.stack(segs)

    def input_to_seg(self, input):
        return self.input3d_to_instance_segmentation(input)


class GenericAdaptedRandError(AdaptedRandError):
    def __init__(
        self, input_channels, thresholds=None, use_last_target=True, ignore_index=None, invert_channels=None, **kwargs
    ):

        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, **kwargs)
        assert isinstance(input_channels, list) or isinstance(input_channels, tuple)
        self.input_channels = input_channels
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        if invert_channels is None:
            invert_channels = []
        self.invert_channels = invert_channels

    def input_to_seg(self, input):
        # pick only the channels specified in the input_channels
        results = []
        for i in self.input_channels:
            c = input[i]
            # invert channel if necessary
            if i in self.invert_channels:
                c = 1 - c
            results.append(c)

        input = np.stack(results)

        segs = []
        for prediction in input:
            for th in self.thresholds:
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label((prediction > th).astype(np.uint8), background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAveragePrecision:
    def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target
        assert metric in ['ap', 'acc']
        if metric == 'ap':
            # use AveragePrecision
            self.metric = AveragePrecision()
        else:
            # use Accuracy at 0.5 IoU
            self.metric = Accuracy(iou_threshold=0.5)

    def __call__(self, input, target):
        if target.dim() == 5:
            if self.use_last_target:
                target = target[:, -1, ...]  # 4D
            else:
                # use 1st target channel
                target = target[:, 0, ...]  # 4D

        input1 = input2 = input
        multi_head = isinstance(input, tuple)
        if multi_head:
            input1, input2 = input

        input1, input2, target = convert_to_numpy(input1, input2, target)

        batch_aps = []
        i_batch = 0
        # iterate over the batch
        for inp1, inp2, tar in zip(input1, input2, target):
            if multi_head:
                inp = (inp1, inp2)
            else:
                inp = inp1

            segs = self.input_to_seg(inp, tar)  # expects 4D
            assert segs.ndim == 4
            # convert target to seg
            tar = self.target_to_seg(tar)

            # filter small instances if necessary
            tar = self._filter_instances(tar)

            # compute average precision per channel
            segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

            logger.info(f'Batch: {i_batch}. Max Average Precision for channel: {np.argmax(segs_aps)}')
            # save max AP
            batch_aps.append(np.max(segs_aps))
            i_batch += 1

        return torch.tensor(batch_aps).mean()

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 0-index
        :param input: input instance segmentation
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    input[input == label] = 0
        return input

    def input_to_seg(self, input, target=None):
        raise NotImplementedError

    def target_to_seg(self, target):
        return target


class BlobsAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BlobsBoundaryAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction, boundary prediction and ground truth instance segmentation.
    Segmentation mask is computed as (P_mask - P_boundary) > th followed by a connected component
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds

    def input_to_seg(self, input, target=None):
        # input = P_mask - P_boundary
        input = input[0] - input[1]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BoundaryAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given boundary prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            seg = measure.label(np.logical_not(input > th).astype(np.uint8), background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class PSNR:
    """
    Computes Peak Signal to Noise Ratio. Use e.g. as an eval metric for denoising task
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return peak_signal_noise_ratio(target, input)


class MSE:
    """
    Computes MSE between input and target
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return mean_squared_error(input, target)


class MultiheadError:
    """
    A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
    (http://brainiac2.mit.edu/SNEMI3D/evaluation).

    This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_seg()`)
    and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
    it's enough to extend this class and implement the `input_to_seg` method.

    Args:
        use_last_target (bool): use only the last channel from the target to compute the ARand
    """

    def __init__(self, use_last_target=False, ignore_index=None, metrics=None, **kwargs):
        self.use_last_target = use_last_target
        self.ignore_index = ignore_index
        self.metrics = metrics

    def __call__(self, inputs, targets):
        """ """

        eval_scores = []
        for metric_dict, input, target in zip(self.metrics, inputs, targets):
            #    ^dict ,  ^list , ^list
            metric_config = metric_dict
            metric_class = _metric_class(metric_config['name'])
            metric = metric_class(**metric_config)
            eval_scores.append(metric(input, target))
        return torch.sum(torch.stack(eval_scores))  # TODO: Maybe, metrics weight?


def _metric_class(class_name):
    m = importlib.import_module('pytorch3dunet.unet3d.metrics')
    clazz = getattr(m, class_name)
    return clazz


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
