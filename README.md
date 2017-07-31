R-FCN
---

This directory contains code to evaluate the R-FCN object detector 
described in the [paper](https://www.robots.ox.ac.uk/~vgg/rg/papers/dai16nips.pdf):

```
"R-fcn: Object detection via region-based fully convolutional networks."  
by Jifeng Dai, Li, Yi, Kaiming He, and Jian Sun (NIPS. 2016).
```

This code is based on the `py-caffe` implementation 
[made available](https://github.com/Orpine/py-R-FCN) by [Yuwen Xiong](https://github.com/Orpine).

The pre-trained models released with the caffe code which have been imported into matconvnet and 
can be downloaded [here](http://www.robots.ox.ac.uk/~albanie/models.html#r-fcn-models).

NOTE: The training code is still in the verfication process.

### Demo

Running the `rfcn_demo.m` script will download a model trained on `pascal voc 2007+2012` data and run it on a sample image to produce the figure below:

<img src="misc/frcn-demo-fig.jpg" width="600" />

### Functionality

There are scripts to evaluate models on the `pascal voc` dataset (the scores produced by the pretrained models are listed on the [model page](http://www.robots.ox.ac.uk/~albanie/models.html#r-fcn-models)).  

### Dependencies

Due to the significant similarity in model design, this code re-uses part of the `mcnFasterRCNN` implementation. The following modules should be added (these can be installed with `vl_contrib`):

* [autonn](https://github.com/vlfeat/autonn) - a wrapper module for matconvnet
* [GPU NMS](https://github.com/albanie/mcnNMS) - a CUDA-based implementation of non-maximum supression
* [mcnFasterRCNN](https://github.com/albanie/mcnFasterRCNN) - MatConvNet Faster R-CNN
  
### Performance

Running the detector with on multiple GPUs produces a significant speed boost during inference.  Timings are shown below for the model based on `ResNet 50`, averaged over a portion of the `pascal 2007` test set using a Tesla M40 GPU (these benchmarks should be considered apprxoimate):

**Multi-GPU code performance:**

| mode      | Single GPU | 2 GPUs   |
|-----------|-----------|-----------|
| inference | 7.5 Hz    | 11 Hz     |
