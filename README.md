# mxnet_osx1011_mbp
A simple guide to install mxnet on a MBP using OSX 10.11

## check your MBP's GPU

Make sure a **NVIDIA discrete GPU card**, e.g., GTX 750M, is on your MBP; w/o it, you only can expect running mxnet on CPU.

## CUDA 7.5 and cuDNN V3

follow official instructions

## compile mxnet C++

On OSX, need update config.mk as follows
- using clang, not gcc
- installing openblas through brew; using apple in the config.mk
- based on [this issue report](https://github.com/dmlc/mxnet/issues/728#issuecomment-160867057), need setup
```bash
ADD_LDFLAGS += -Xlinker -F/Library/Frameworks -Xlinker -framework -Xlinker CUDA
```

Copy the provided **config.mk** file to the root-dir of the mxnet and then run make command


## install mxnet-R package
- follow [this instruction](http://apple.stackexchange.com/questions/208478/how-do-i-disable-system-integrity-protection-sip-aka-rootless-on-os-x-10-11)  to disable SIP (aka. rootless) on OSX 10.11
- make sure R is > 3.2
- have these path setups
```bash
# 12/2/2015
# for mxnet-R
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
export DYLD_FALLBACK_LIBRARY_PATH=.:/usr/local/cuda/lib:$DYLD_FALLBACK_LIBRARY_PATH
```
- using installation from the source
- for **RStudio**, please start it from terminal so that the mxnet package can be loaded

Using the provided **lenet.R** to test running on both CPU and GPU
