### run training from scratch
```
python ./drn/segment.py train -d '/home/patrick/disk1/CAN/data' -c 4 -s 256 \
    --arch drn_d_22 --batch-size 8 --epochs 100 --lr 0.01 --momentum 0.9 \
    --step 100
```
### resume training
```
python ./drn/segment.py train -d '/home/patrick/disk1/CAN/data' -c 4 -s 256 \
    --arch drn_d_22 --batch-size 8 --epochs 100 --lr 0.01 --momentum 0.9 \
    --step 100 --resume '/home/patrick/disk1/CAN/checkpoint_latest.pth.tar'
```
### test on validation set
```
python ./drn/segment.py test -d '/home/patrick/disk1/CAN/data' -c 4 \
    --arch drn_d_22 --batch-size 1 --resume '/home/patrick/disk1/CAN/model_best.pth.tar' --phase val
```


