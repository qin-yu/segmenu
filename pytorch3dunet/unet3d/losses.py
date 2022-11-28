import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

from pytorch3dunet.unet3d.utils import get_logger, expand_as_one_hot

logger = get_logger('Loss')


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # flatten (N, C, D, H, W) -> (C, N * D * H * W)
    input = per_channel_flatten(input)
    target = per_channel_flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        if isinstance(target, torch.Tensor):
            assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'
            target = target[:, :-1, ...]  # skips last target channel if needed
            if self.squeeze_channel:  # squeeze channel dimension if singleton
                target = torch.squeeze(target, dim=1)
        else:
            assert target[0].size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'
            target = [this_target[:, :-1, ...] for this_target in target]
            if self.squeeze_channel:  # squeeze channel dimension if singleton
                target = [torch.squeeze(this_target, dim=1) for this_target in target]
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, normalization='sigmoid'):
        super().__init__()
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x


class _AbstractSingleDiceLoss(_AbstractDiceLoss):
    """
    Base class for single head implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__()
        self.register_buffer('weight', weight)

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1.0 - torch.mean(per_channel_dice)


class _AbstractMultiDiceLoss(_AbstractDiceLoss):
    """
    Base class for multi-head implementations of Dice loss.
    """

    def __init__(self, weights=None, normalization='sigmoid'):
        super().__init__()
        self.register_buffer('weights', weights)


class DiceLoss(_AbstractSingleDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def dice(self, input, target):
        # NOTE: self.dice() is just compute_per_channel_dice(), but self.forward() is 1.0 - dice()
        return compute_per_channel_dice(input, target, weight=self.weight)


class MultiheadDiceLoss(_AbstractMultiDiceLoss):
    """Computes Dice Loss on each head according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.

    This does not inherit from `DiceLoss` because multi-head networks take a list of weight arrays, thus `weights`
    """

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight)

    def forward(self, inputs, targets):
        """`inputs` and `targets` are lists with head elements each of shape (N, C, D, H, W).
        For a 2-head network with 2-channel output, `inputs` is [(N, C, D, H, W), (N, C, D, H, W)].
        Here, `self.weight` must also be per channel and per head, i.e. [(h1c1, h1c2), (h2c1, h2c2)].
        """
        # get probabilities from logits
        inputs = [self.normalization(input) for input in inputs]

        # compute per channel Dice coefficient
        per_channel_dice_list = [
            self.dice(input, target, weight=weight) for input, target, weight in zip(inputs, targets, self.weights)
        ]

        # average Dice score across all channels/classes
        losses = [1.0 - torch.mean(per_channel_dice) for per_channel_dice in per_channel_dice_list]

        return torch.sum(torch.stack(losses))


class CrossHeadDiceLoss(nn.Module):
    """
    Linear combination of Cross-head Dice and Dice losses
    """

    def __init__(self, weights, head_0_channel, head_1_channel, normalization='sigmoid', c=1.0):
        super().__init__()
        # self.register_buffer('c', c)  # cross-head dice coefficient
        self.c = c  # cross-head dice coefficient
        self.multidice = MultiheadDiceLoss(weights=weights)
        self.dice = DiceLoss(normalization=normalization)
        self.channels = [head_0_channel, head_1_channel]

    def forward(self, inputs, targets):
        if len(inputs) > 2:
            raise ValueError("CrossHeadDiceLoss accepts only two predictions/heads.")
        cell_boundary = inputs[0][:, self.channels[0], :, :, :]
        nuclei_foreground = inputs[1][:, self.channels[1], :, :, :]

        loss_cross_head = 1.0 - self.dice(cell_boundary, nuclei_foreground)  # just a per-channel dice
        loss = self.multidice(inputs, targets) + self.c * loss_cross_head
        return loss


class ProductLoss(nn.Module):  # failure: need more exploration.
    """Sum of element-wise multiplications"""

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        prob1 = F.softmax(input1)
        prob2 = F.softmax(input2)
        return (prob1 * prob2).mean()


class ProductLossNew(nn.Module):  # not finished
    """Sum of element-wise multiplications"""

    def __init__(self):
        super().__init__()
        self.normalization = nn.Sigmoid()

    def forward(self, input1, input2):
        prob1 = self.normalization(input1)
        prob2 = self.normalization(input2)
        return (prob1 * prob2).mean()


class DotDiceLoss(nn.Module):  # failure
    """
    Linear combination of Dot Product and Dice losses

    This ruins the training because `(cell_boundary * nuclei_foreground).mean()` grows wild too.
    """

    def __init__(self, weights, normalization='sigmoid', c=1.0):
        super().__init__()
        # self.register_buffer('c', c)  # cross-head dice coefficient
        self.c = c  # cross-head dice coefficient
        self.multidice = MultiheadDiceLoss(weights=weights)
        self.product = ProductLoss()

    def forward(self, inputs, targets):
        if len(inputs) > 2:
            raise ValueError("CrossHeadDiceLoss accepts only two predictions/heads.")
        cell_boundary = inputs[0][:, 0, :, :, :]
        nuclei_foreground = inputs[1][:, 0, :, :, :]
        value1 = self.multidice(inputs, targets)
        value2 = self.product(cell_boundary, nuclei_foreground)
        # print(value1, value2)
        loss = value1 + self.c * value2
        logger.info(f"Multihead dice loss: {value1}, product loss: {value2}, effective product loss {self.c * value2}")
        return loss


class DynamicDotDiceLoss(nn.Module):  # not finished
    """
    Linear combination of Dot Product and Dice losses

    This ruins the training because `(cell_boundary * nuclei_foreground).mean()` grows wild too.
    """

    def __init__(self, weights, normalization='sigmoid'):
        super().__init__()
        # self.register_buffer('c', c)  # cross-head dice coefficient
        # if c is not None and dynamic is not None:
        #     raise ValueError("Cannot have fixed weight and dynamic weight at the same time")
        # self.c = c  # cross-head dice coefficient
        self.multidice = MultiheadDiceLoss(weights=weights)
        self.product = ProductLoss()

    def forward(self, inputs, targets):
        if len(inputs) > 2:
            raise ValueError("CrossHeadDiceLoss accepts only two predictions/heads.")
        cell_boundary = inputs[0][:, 0, :, :, :]
        nuclei_foreground = inputs[1][:, 0, :, :, :]
        value1 = self.multidice(inputs, targets)
        value2 = self.product(cell_boundary, nuclei_foreground)
        # print(value1, value2)
        loss = value1 + self.c * value2
        logger.info(f"Multihead dice loss: {value1}, product loss: {value2}, effective product loss {self.c * value2}")
        return loss


class DotGTDiceLoss(nn.Module):
    """
    Linear combination of Dot Product and Dice losses
    """

    def __init__(self, weights):
        super().__init__()
        self.multidice = MultiheadDiceLoss(weights=weights)

    def forward(self, inputs, targets):
        if len(inputs) > 2:
            raise ValueError("CrossHeadDiceLoss accepts only two predictions/heads.")
        cell_boundary = inputs[0][:, 1, :, :, :]
        cell_boundary_gt = targets[0][:, 1, :, :, :]
        nuclei_foreground = inputs[1][:, 0, :, :, :]
        nuclei_foreground_gt = targets[1][:, 0, :, :, :]
        value1 = self.multidice(inputs, targets)
        value2 = (cell_boundary * nuclei_foreground_gt).mean()
        value3 = (nuclei_foreground * cell_boundary_gt).mean()
        print(value1, value2, value3)
        loss = value1 + 0.5 * value2 + 0.5 * value3
        return loss


class GeneralizedDiceLoss(_AbstractSingleDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf."""

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = per_channel_flatten(input)
        target = per_channel_flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf"""

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = per_channel_flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(1)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def per_channel_flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)
    weights = loss_config.pop('weights', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    if weights is not None:
        # convert to cuda tensor if necessary
        weights = torch.tensor(weights).to(config['device'])

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config['device'])

    loss = _create_loss(name, loss_config, weight, weights, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    return loss


#######################################################################################################################


def _create_loss(name, loss_config, weight, weights, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alphs', 1.0)
        beta = loss_config.get('beta', 1.0)
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == 'MultiheadDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return MultiheadDiceLoss(weights=weights, normalization=normalization)
    elif name == 'CrossHeadDiceLoss':
        head_0_channel = loss_config.get('head_0_channel')
        head_1_channel = loss_config.get('head_1_channel')
        normalization = loss_config.get('normalization', 'sigmoid')
        cross_head_dice_coef = loss_config.get('c', 1.0)
        return CrossHeadDiceLoss(weights=weights, head_0_channel=head_0_channel, head_1_channel=head_1_channel, normalization=normalization, c=cross_head_dice_coef)
    elif name == 'DotDiceLoss':
        cross_head_dice_coef = loss_config.get('c', None)
        normalization = loss_config.get('normalization', 'sigmoid')
        return DotDiceLoss(weights=weights, normalization=normalization, c=cross_head_dice_coef)
    elif name == 'DotGTDiceLoss':
        return DotGTDiceLoss(weights=weights)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(
            threshold=loss_config['threshold'],
            initial_weight=loss_config['initial_weight'],
            apply_below_threshold=loss_config.get('apply_below_threshold', True),
        )
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
