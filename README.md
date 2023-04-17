# SVRP

This is the code for our ECCV 2022 paper "Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning".

![image](https://github.com/ht014/svrp/blob/main/framework.png)
## Installation
#### Pytorch
```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
```
#### CLIP
```
$ Please follow the official [instructrions](https://github.com/openai/CLIP) to install CLIP.
```
#### maskrcnn
```
$ Check [INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/INSTALL.md) to install maskrcnn. Then, adding the maskrcnn lib to your $PYTHONPATH, because our code uses the ROIAlign layer to extract the roi features.
```
## Pretraining and finetuning
```
$ sh run_pretrain.sh # pretrain the visual relationship model. 
$ sh run.sh # finetuning the model.
```
## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@InProceedings{he2021exploiting,
    author    = {He, Tao and Gao, Lianli and Song, Jingkuan and Li, Yuan-Fang},
    title     = {Exploiting Scene Graphs for Human-Object Interaction Detection},
    booktitle = {International Conference on Computer Vision(ICCV)},
    year      = {2021},
    url       = {https://arxiv.org/pdf/2108.08584}
}
```
## Acknowledgement

This repository is developed on top of the other two projects: TDE by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and CLIP by [Openai](https://github.com/openai/CLIP). 
