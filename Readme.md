#Steel detection with context aggregated network

credit to https://github.com/fyu/drn

##Proprocess

1. Download the data from kaggle and put it in the `./data` directory, unzip the `train_images.zip` to `./data/img`
2. Run in bash:
`
cd data
python ./preprocess.py
./create_list.sh
python ./produce_info_json.py
`
This would create a directory `./data/mask`. The labels are stored in this directory.
Also, there will be 4 text files in `./data`, each containing the path to images/labels.
The `produce_info_json.py` writes the mean and standard deviation of the images to `info.json`. This is already provided.

##Train
Refer to script.md
