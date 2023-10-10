# Vision Transformer

## Introduction
This is an implementation of Visional Transformer from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) using Pytorch. The model is tested on the MNIST dataset and the Cifar-10 dataset.

The model is trained for 10 epochs on the MNIST dataset, and 150 epochs on the Cifar-10 dataset.



## Result

The results are obtained using the default hyperparameters in the `train.py` file. The only modification is the patch size (`-p`). To ensure the image height and weight are divisible by the patch size, you are required to specify a suitable `-p`, otherwise an error will be thrown. 

|     |test acc |
|-----|--------|
|MNIST|0.9795     |
|Cifar-10|0.8061      |