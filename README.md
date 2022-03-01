# OSSL
## PyTorch Implementation

This repository contains the example on CIFAR-10 setup 

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux 
* NVIDIA GPU + CUDA CuDNN 
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/data/```)
* [CIFAR10.1](https://github.com/modestyachts/CIFAR-10.1) (downloaded and unzipped to ```PROJECT_DIR/data/CIFAR-10.1```)
* Please use PyTorch1.5 to avoid compilation errors (other versions should be good)
* You might need to change the file paths, and please be sure you change the corresponding paths in the codes as well     

## Getting started
1. Clone the repository using the following command
```
git clone https://github.com/harshita-555/OSSL.git
```
2. Change your working directory to OSSL/
```
cd OSSL
```
3. Learn classifier on CIFAR-10 (DenseNet-10-12)
  ```bash
    # Save as "PROJECT_DIR/DenseNet-40-12-ss/checkpoint.pth.tar"
    # Modified based on the wonderful github of https://github.com/andreasveit/densenet-pytorch
    python train.py --layers 40 --growth 12 --no-bottleneck --reduce 1.0 --epochs 250 --rot --two_classifiers 
 ```
        

## License
MIT

## Reference

The implementation reused some portions from [Accuracy-estimation-with-self-supervision](https://github.com/Simon4Yan/Accuracy-estimation-with-self-supervision.git)[1].


1. W. Deng, S. Gould, and L. Zheng, “What does rotation prediction tell us about classifier accuracy under varying testing environments?” in International Conference on Machine Learning. PMLR, 2021, pp.2579–2589.
