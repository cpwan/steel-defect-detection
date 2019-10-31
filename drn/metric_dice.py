import torch

def dice_channel_torch(probability, truth, threshold):
    """	
    credit: https://www.kaggle.com/wh1tezzz/correct-dice-metrics-for-this-competition
    """
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = torch.Tensor([0.]).cuda()
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                t=truth[i, j, :, :]
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel

def dice_single_channel(probability, truth, threshold, eps = 1E-9):
    """	
    credit: https://www.kaggle.com/wh1tezzz/correct-dice-metrics-for-this-competition
    """
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() +eps)/ (p.sum() + t.sum() + eps)
    return dice

def dice(output, target):
    def one_hot(x, class_count):
        return torch.eye(class_count)[x,:]
    nclasses=output.shape[1]
    target_=target.clone().detach()
    target_[target_==255]=nclasses
    one_hot_target = torch.nn.functional.one_hot(target_, nclasses+1 )[:,:,:,0:nclasses].permute(0,3,1,2)
    out = dice_channel_torch(output[:,0:nclasses,:,:],one_hot_target,0.5)
    return out.item()



def dice_loss(output, target, weight=None, ignore_index=None, reduction='mean'):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    ---
    credit: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
    """
    eps = 0.0001

    #output = output.exp()
    encoded_target = output.detach() * 0
    if ignore_index is not None:
        mask = target == ignore_index
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

#    if weight is None:
#        weight = 1

#    intersection = output * encoded_target
#    numerator = 2 * intersection.sum(0).sum(1).sum(1)
#    denominator = output + encoded_target

#    if ignore_index is not None:
#        denominator[mask] = 0
#    denominator = denominator.sum(0).sum(1).sum(1) + eps
#    dice_per_channel = weight *  (numerator / denominator)
#    dice_avg=dice_per_channel.sum() / output.size(1)
    bceLoss=torch.nn.functional.binary_cross_entropy(output, encoded_target)
    #return loss_avg+bceLoss
    return bceLoss # returning BCELoss instead, empirically BCELoss alone gives a better traning


class DICELoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'weight', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(DICELoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return dice_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
