# mxnet_osx1011_mbp
A simple guide to install mxnet on a MBP using OSX 10.11

## check your MBP's GPU
make sure a NVIDIA discrete GPU card is on your MBP; w/o it, you only can expect running mxnet on CPU

## CUDA 7.5 and cuDNN V3

## compile mxnet C++

Keep in mind,
- using clang, not gcc
- blas; using Apple

Copy the provided config.mk file to the root-dir of the mxnet and then

```bash
make -j4
```

## install mxnet-R package 
- follow this to disable
- make sure R is > 3.2
- LD_LIBRARY_PATH
- using installation from the source
- for **RStudio**, please start it from terminal so that the mxnet package can be loaded

Using lenet.R to test running on both CPU and GPU
