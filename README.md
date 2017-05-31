# [cknowledge.org/ai](http://cknowledge.org/ai): Crowdsourcing benchmarking and optimisation of AI

A suite of open-source tools for [collecting knowledge on optimising AI](http://bit.ly/hipeac49-ckdl):
* [Android app](https://play.google.com/store/apps/details?id=openscience.crowdsource.video.experiments&hl=en_GB)
* [Desktop app](https://github.com/dividiti/ck-crowdsource-dnn-optimization)
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-Caffe2](https://github.com/ctuning/ck-caffe2)
* [CK-TensorFlow](https://github.com/ctuning/ck-tensorflow)
* [CK-TensorRT](https://github.com/dividiti/ck-tensorrt)
* [CK-KaNN](https://github.com/dividiti/ck-kann)
* etc.

# [PUBLIC] Benchmarking Caffe on Firefly-RK3399

The Jupyter notebook (
[view on github.com](https://github.com/dividiti/ck-caffe-firefly-rk3399/blob/master/script/batch_size-libs-models/analysis.20170531.ipynb);
[view on nbviewer.jupyter.org](https://nbviewer.jupyter.org/github/dividiti/ck-caffe-firefly-rk3399/blob/master/script/batch_size-libs-models/analysis.20170531.ipynb)
) in this Collective Knowledge repository analyses the execution time of
inference (forward propagation):

- on the [Firefly-RK3399](http://en.t-firefly.com/en/firenow/Firefly_RK3399) development board:
  - Chip:
    - [Rockchip RK3399](http://rockchip.wikidot.com/rk3399)
  - CPU ("big"):
    - ARM&reg; Cortex&reg;-A72 architecture
    - Max clock 1800 MHz;
    - 2 cores;
  - CPU ("LITTLE"):
    - ARM&reg; Cortex&reg;-A53 architecture;
    - Max clock 1416 MHz;
    - 4 cores;
  - GPU:
    - ARM&reg; Mali&trade;-T860 architecture;
    - Max clock 800 MHz;
    - 4 cores;
    - OpenCL driver:
```
$ ck run program:tool-print-opencl-devices | grep "version:"
OpenCL 1.2 v1.r13p0-00rel0-git(a4271c9).1ad3fc5b648bfed782705ad469a9797a
```

  - RAM:
    - Samsung dual-channel DDR3;
    - 4 GB;
  - BSP:
    - [Firefly-rk3399_xubuntu1604_201704100952_Beta.7z](http://bbs.t-firefly.com/forum.php?mod=viewthread&tid=1876&extra=page%3D1)
```
$ cat /etc/lsb-release 
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=16.04
DISTRIB_CODENAME=xenial
DISTRIB_DESCRIPTION="Ubuntu 16.04.2 LTS"
$ uname -a
Linux firefly 4.4.52 #9 SMP Fri Mar 24 10:53:56 HKT 2017 aarch64 aarch64 aarch64 GNU/Linux
```

- of the following **Caffe models**:

```
$ ck show env --tags=caffemodel
Name:                                                      Version:                 Tags:

Caffe model (net and weights) (deepscale, squeezenet, 1.1) 1.1      32bits,bvlc,caffe,caffemodel,deepscale,host-os-linux-32,net,squeezenet,target-os-linux-32,v1,v1.1,weights

Caffe model (net and weights) (bvlc, googlenet)            trunk    32bits,bvlc,caffe,caffemodel,googlenet,host-os-linux-32,mirror,net,target-os-linux-32,v0,weights

Caffe model (net and weights) (bvlc, alexnet)              trunk    32bits,alexnet,bvlc,caffe,caffemodel,host-os-linux-32,mirror,net,target-os-linux-32,v0,weights
```

- with the following **Caffe libraries**:

```
$ ck show env --tags=lib,caffe
Name:                                             Version:       Tags:

BVLC Caffe framework (opencl,clblast,tune) trunk-880111e 32bits,bvlc,caffe,host-os-linux-32,lib,target-os-linux-32,v0,v0.0,vclblast,vopencl,vtune

BVLC Caffe framework (opencl,clblast)      trunk-880111e 32bits,bvlc,caffe,host-os-linux-32,lib,target-os-linux-32,v0,v0.0,vclblast,vopencl

BVLC Caffe framework (cpu)                 trunk-eeebdab 32bits,bvlc,caffe,host-os-linux-32,lib,target-os-linux-32,v0,v0.0,vcpu,vmaster

...
```

- built with the following **BLAS libraries**:

```
$ ck show env --tags=lib,blas
Name:            Version:            Tags:

clBLAS library   2.10-d16f7b3        32bits,blas,clblas,host-os-linux-32,lib,opencl-blas,target-os-linux-32,v2,v2.10,v2.10.0

ViennaCL library dvdt-master-5494bef 32bits,blas,host-os-linux-32,lib,opencl-blas,src,target-os-linux-32,v0,v0.0,v0.0.0,vdvdt,viennacl,vsrc

ViennaCL library master-5494bef      32bits,blas,host-os-linux-32,lib,opencl-blas,target-os-linux-32,v0,v0.0,vcpu,viennacl,vmaster

ViennaCL library master-5494bef      32bits,blas,host-os-linux-32,lib,opencl-blas,src,target-os-linux-32,v0,v0.0,viennacl,vmaster,vsrc

OpenBLAS library 0.2.19-85636ff      32bits,blas,cblas,host-os-linux-32,lib,openblas,target-os-linux-32,v0,v0.2,v0.2.19,v0.2.19.0

CLBlast library  development-3ec14df 32bits,blas,clblast,host-os-linux-32,lib,no-tune,opencl-blas,target-os-linux-32,v0,v0.0,vdev,vmaster

CLBlast library  development-c9f39ed 32bits,blas,clblast,clblast-tune,host-os-linux-32,lib,opencl-blas,target-os-linux-32,tune,v0,v0.0,vdev,vmaster

```

- with the **batch size** varying from 2 to 16 with step 2.
