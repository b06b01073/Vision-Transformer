# Vision Transformer

## Introduction
This is an implementation of Visional Transformer from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) using Pytorch. The model is tested on the MNIST dataset and the Cifar-10 dataset.

The model is trained until it stop improving on the test set for a consecutive `--patience` epochs (`patience=10` for the result below), or the training reaches the maximum training epochs.



## Result

The results are obtained using the default hyperparameters in the `train.py` file. The only modification is the patch size (`-p`). To ensure the image height and weight are divisible by the patch size, you are required to specify a suitable `-p`, otherwise an error will be thrown (we use `-p=7` for MNIST and `-p=8` for Cifar10). 

If you want to compare the performance yourself and the channel of the input image is not 3, then you are required to modify the `VisionTransformer` class in `torchvision.models.vision_transformer` by adding an addtional `in_channels` argument. 


### Test accurary


|     | This repo | Pytorch official's implementation |
|-----|--------|------|
|MNIST|0.9830     |0.9702|
|Cifar-10|0.7860    | 0.7576

> The result is by no means to tell my implementation is better than or worse than the Pytorch official's implementation. This result only suggests that the implementation is highly possible to be correct (there's no bugs that makes this model performs poorly, I mean, I hope so).

![akward](https://media.tenor.com/AFJsOL0Ox-AAAAAC/awkward.gif)