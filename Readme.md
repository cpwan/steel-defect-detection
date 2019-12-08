# Steel defect detection with U-net and CAN

reference: 

> [U-net:   Convolutional  networks  for  biomedical  image  segmentation](https://arxiv.org/abs/1505.04597)

> [Dilated Residual Networks](https://arxiv.org/abs/1705.09914)

Code structure modified on top of https://github.com/fyu/drn

## Proprocess

1. Download the data from kaggle and put it in the `./data` directory, unzip the `train_images.zip` to `./data/img`

2. Run in bash:
```
cd data
python ./preprocess.py
./create_list.sh
python ./produce_info_json.py
```
This would create a directory `./data/mask`. The labels are stored in this directory.

Also, there will be 4 text files in `./data`, each containing the path to images/labels.

The `produce_info_json.py` writes the mean and standard deviation of the images to `info.json`. This is already provided.

## Training/ eavaluating
Refer to [script.md](https://github.com/cpwan/steel-defect-detection/blob/drn/script.md)


