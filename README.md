# Efficient-Ensemble

This repository is for STAT433(Statistical Modeling for Deep Learning) Term Project.

# How to run
```
python train.py --ensemble-num [num] --model [simplecnn/resnet20/resnet32] --dataset [cifar10/cifar100] --epoch [num]
ex) python train.py --ensemble-num 1 --model resnet20 --dataset cifar10 --epoch 50
```

# code reference
* https://github.com/akamaster/pytorch_resnet_cifar10
* https://github.com/pytorch/examples/blob/master/mnist/main.py
* https://github.com/victoresque/pytorch-template
* https://github.com/noowad93/MIMO-pytorch