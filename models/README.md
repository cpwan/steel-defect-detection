# Credit
The codes are modified on top of https://github.com/fyu/drn.

# Changes w.r.t. the original implementation
- added dice loss and dice score in `metric_dice.py`
- modification to `segment.py`
- added `allocateGPU.py` for allocating at most 1 GPU


#### modification to `segment.py`:




model design:
- Replaced the last layer to `Sigmoid` from `LogSoftmax` (Now the model report the probability, independent of channel)
- Replaced the criterion to `metric_dice.DICELoss` instead of `NLLLoss`
- Replaced the decision rule: thresholding with 0.5 when taking argmax in channels
- Modified the `accuracy` method to be a wrapper of `metric_dice.dice` instead of really reporting accuracy 

miscellaneous:
- Accomodated syntax to torch 1.1.0
- Adapted the color palatte to 4 colors for types of steel defects
- Removed RandomCrop in DataLoader, using the whole image instead in training stage
- Set pin_memory to False in DataLoaders to avoid using too much CPU.

