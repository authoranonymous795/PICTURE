

<div align=center>
<img src="assets/PICTURE2_green.png" width="210px">
</div>


 <h3 align="center"><strong>Point Cloud Reconstruction Is Insufficient to Learn 3D Representations</strong></h3>

------

  <p align="center">
    <strong>Anonymous author</strong></a>&nbsp;&nbsp;&nbsp;
    <br>
    Affiliation&nbsp;&nbsp;&nbsp;<br>
    email@example.com
  </p>
  <table align="center">
    <tr>
    <td>
      <img src="assets/overview_480p_speed.gif">
    </td>
    </tr>
  </table>
  <p align="center">
    <strong>This GitHub repository and all external links have been anonymized!</strong></a>&nbsp;&nbsp;&nbsp;
  </p>



## :books:Outline

- [Location of Key Codes](#sparkles-location-of-key-codes)
- [Main Results](#car-main-results)
- [Getting Start](#%EF%B8%8Fgetting-start)
  - [1. Download Weights of MinkUNet (Res16UNet34C) Pre-trained by Seal](#%EF%B8%8F1-download-weights-of-minkunet-res16unet34c-pre-trained-by-seal)
  - [2. Prepare Dataset](#%EF%B8%8F2-prepare--dataset)
  - [3. Prepare the Environment](#%EF%B8%8F3-prepare-the-environment)
  - [4. Prepare the Seal Feature for the Entire Dataset Offline](#%EF%B8%8F4-prepare-the-seal-feature-for-the-entire-dataset-offline)
  - [5. Run the Code](#rocket5-run-the-code)



## :sparkles: Location of Key Codes

In [pcdet/models/dense_heads/pretrain_head_3D_seal.py](pcdet/models/dense_heads/pretrain_head_3D_seal.py), we provide the implementations of `3D High-level Voxel Feature Generation Module`, which involves the processes of data extraction, voxelization, as well as the computation of the target and loss.

In [pcdet/models/backbones_3d/I2Mask.py](pcdet/models/backbones_3d/I2Mask.py) and [pcdet/models/backbones_3d/dsvt_backbone_mae.py](pcdet/models/backbones_3d/dsvt_backbone_mae.py), we provide the implementations of  `Inter-class and Intra-class Discrimination-guided Masking` (I2Mask).

We provide all the configuration files in the paper and appendix in [tools/cfgs/picture_models/](tools/cfgs/picture_models/). 

👆 [BACK to Table of Contents -->](#booksoutline)

 

## :car: Main Results

**Pre-training**

Waymo

| Model                                                        | Pre-train Fraction |                       Pre-train model                        |                             Log                              |
| ------------------------------------------------------------ | :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [PICTURE (DSVT)](tools/cfgs/picture_models/picture_waymo_ssl_seal_decoder_mask_0.2.yaml) |        20%         | [ckpt](https://www.dropbox.com/scl/fi/xqophis4ay2ukj0owocl8/pretrain_dsvt_waymo_0.2.pth?rlkey=47tdx4wntnbrzzddmbesbblug&dl=0) | [Log](https://www.dropbox.com/scl/fi/vit0k7mtcpf8yjjj61v0f/log_train_20231102-141005-pretrain_waymo_0.2.txt?rlkey=47w87gkgvaeaegtymhj29lk66&dl=0) |
| [PICTURE (DSVT)](tools/cfgs/picture_models/picture_waymo_ssl_seal_decoder_mask.yaml) |        100%        | [ckpt](https://www.dropbox.com/scl/fi/uauvxy2j63rwdhucoanc5/pretrain_dsvt_waymo.pth?rlkey=njfzulqueuw7vgciawzsx7vjw&dl=0) | [Log](https://www.dropbox.com/scl/fi/qli12eq5gusqoehwsxwxl/log_train_20231121-082115-pretrain_waymo.txt?rlkey=cq5pt8dsp70qa5hum6nlzs01p&dl=0) |

nuScenes

| Model                                                        |                       Pre-train model                        |                             Log                              |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [PICTURE (DSVT)](tools/cfgs/picture_models/picture_nuscenes_ssl_seal_decoder_mask.yaml) | [ckpt](https://www.dropbox.com/scl/fi/qsinuu476c5s2n47vhpmu/pretrain_dsvt_nuscenes.pth?rlkey=3ddq9nbodh7lhf2z2bjrieal9&dl=0) | [Log](https://www.dropbox.com/scl/fi/zcecxh9ikh6p4u71amppr/log_train_20231205-105152-pretrain_nuscenes.txt?rlkey=o3dmn23lba41nvv90kb88izcw&dl=0) |



**Fine-tuning**

3D Object Detection (on Waymo validation)

| Model                                                        | Pre-train Fraction |  mAP/H_L2   |   Veh_L2    |   Ped_L2    |   Cyc_L2    |                             ckpt                             |                             Log                              |
| ------------------------------------------------------------ | :----------------: | :---------: | :---------: | :---------: | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DSVT (PICTURE)](tools/cfgs/picture_models/picture_waymo_detection_0.2.yaml) |        20%         | 73.84/71.80 | 71.66/71.35 | 75.88/70.52 | 73.98/73.53 | [ckpt](https://www.dropbox.com/scl/fi/8dv1az23n057p8aicw0wg/finetune_dsvt_waymo_detection_0.2.pth?rlkey=gh3d8vy4fiai7eu8cnhfoo7uc&dl=0) | [Log](https://www.dropbox.com/scl/fi/cfkbtzoxz1ywnpgf7036h/log_train_20231114-133337-finetune_waymo_detection_0.2.txt?rlkey=8qorp5k0el6t9up9kvk4e1j29&dl=0) |
| [DSVT (PICTURE)](tools/cfgs/picture_models/picture_waymo_detection.yaml) |        100%        | 75.13/72.69 | 72.93/72.45 | 77.18/71.66 | 75.27/73.96 | [ckpt](https://www.dropbox.com/scl/fi/6f9hjqhy6qlbf7s7g0py2/finetune_dsvt_waymo_detection.pth?rlkey=pup56t2luhr2pzis1gt7ow19n&dl=0) | [Log](https://www.dropbox.com/scl/fi/7gfbjqcmy2911wc2h99hh/log_train_20231126-093422-finetune_waymo_detection.txt?rlkey=rsuhrku6c1n5m46aec80y2ztp&dl=0) |

3D Object Detection (on nuScenes validation)

| Model                                                        | mAP  | NDS  | mATE | mASE | mAOE | mAVE | mAAE |                             ckpt                             |                             Log                              |
| ------------------------------------------------------------ | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DSVT (PICTURE)](tools/cfgs/picture_models/picture_nuscenes_detection.yaml) | 68.1 | 72.6 | 25.8 | 24.2 | 26.5 | 21.6 | 17.7 | [ckpt](https://www.dropbox.com/scl/fi/cpb81ic9evn81shdwvvce/finetune_dsvt_nuscenes_detection.pth?rlkey=vczq90q33z0g1xq57hq7k69re&dl=0) | [Log](https://www.dropbox.com/scl/fi/c87nd9e39nqp0e6fdola5/log_train_20231212-154208-finetune_nuscenes_detection.txt?rlkey=jfmlduaag3a3wf17k4fiplyq3&dl=0) |

3D Semantic Segmentation (on nuScenes validation)

| Model                                                        | mIoU | bicycle | bus  | car  | motorcycle | pedestrian | trailer | truck |                             ckpt                             |                             Log                              |
| ------------------------------------------------------------ | :--: | :-----: | :--: | :--: | :--------: | :--------: | :-----: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [Cylinder3D-SST (PICTURE)](tools/cfgs/picture_models/picture_nuscenes_segmentation.yaml) | 79.4 |  43.2   | 94.5 | 96.3 |    80.6    |    84.1    |  65.7   | 87.5  | [ckpt](https://www.dropbox.com/scl/fi/wt7q0yoykknj7t3yyc2wj/finetune_dsvt_nuscenes_segmentation.pth?rlkey=jwjuz45qfg1bkppquo9fswyx5&dl=0) | [Log](https://www.dropbox.com/scl/fi/ay30hiy39uzvi57vi55cd/log_train_20231216-092123-finetune_nuscenes_segmentation.txt?rlkey=oykxuhe1xfs0mnsykbaqrmjg8&dl=0) |

Occupancy Prediction (on nuScenes OpenOccupancy validation)

| Model                                                        | mIoU | bicycle | bus  | car  | motorcycle | pedestrian | trailer | truck |                             ckpt                             |                             Log                              |
| ------------------------------------------------------------ | :--: | :-----: | :--: | :--: | :--------: | :--------: | :-----: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DSVT (PICTURE)](tools/cfgs/picture_models/picture_nuscenes_occupancy.yaml) | 18.4 |   8.1   | 15.6 | 20.7 |    7.4     |    12.3    |  15.7   | 15.8  | [ckpt](https://www.dropbox.com/scl/fi/wn6cvusit5hb5pvaafbzt/finetune_dsvt_nuscenes_occupancy.pth?rlkey=l0qj886c5ofqmc4hbdj6wf8p8&dl=0) | [Log](https://www.dropbox.com/scl/fi/6uwctfld98w36u2uw1o2i/log_train_20231221-142502-finetune_nuscenes_occupancy.txt?rlkey=gs2v5yds2xkf1tg08g2g1nnpa&dl=0) |

👆 [BACK to Table of Contents -->](#booksoutline)



## 🏃‍♂️Getting Start

### ⬇️1. Download Weights of MinkUNet (Res16UNet34C) Pre-trained by Seal

[youquanl/Segment-Any-Point-Cloud: NeurIPS'23 Spotlight\] Segment Any Point Cloud Sequences by Distilling Vision Foundation Models (github.com)](https://github.com/youquanl/Segment-Any-Point-Cloud)

After downloading, please put it into project path

👆 [BACK to Table of Contents -->](#booksoutline)



### ⚒️2. Prepare  Dataset

**Waymo**：

1.Download the Waymo dataset from the [official Waymo website](https://waymo.com/open/download/), and make sure to download version 1.2.0 of Perception Dataset.

2.Prepare the directory as follows:

```
PICTURE
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
├── pcdet
├── tools
```

3.Prepare the environment and install `waymo-open-dataset`:

```
pip install waymo-open-dataset-tf-2-5-0
```

4.Generate the complete dataset. It require approximately 1T disk and 100G RAM.

```shell
# only for single-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

# for single-frame or multi-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
# Ignore 'CUDA_ERROR_NO_DEVICE' error as this process does not require GPU.
```



**nuScenes**：

1.Prepare the `trainval` dataset from [nuScenes](https://www.nuscenes.org/nuscenes#download) and prepare the directory as follows:

```shell
PICTURE
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

2.Prepare the environment and install `nuscenes-devkit`：

```
pip install nuscenes-devkit==1.0.5
```

3.Generate the complete dataset.

```shell
# for lidar-only setting
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
```



**nuScenes Lidarseg**：

1.Download the annotation files from [nuScenes](https://www.nuscenes.org/nuscenes#download) and prepare the directory as follows:

```
PICTURE
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │   │── lidarseg.json
│   │   │   │   │── category.json
│   │   │── lidarseg
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```



**nuScenes OpenOccupancy**：

1.Download the annotation files from [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) and prepare the directory as follows:

```
PICTURE
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │   │── lidarseg.json
│   │   │   │   │── category.json
│   │── nuScenes-Occupancy
├── pcdet
├── tools
```

2.Prepare the environment:

```shell
conda install -c omgarcia gcc-6 # gcc-6.2
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# Install mmdet3d from source code.
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install

# Install occupancy pooling.
git clone https://github.com/JeffWang987/OpenOccupancy.git
cd OpenOccupancy
export PYTHONPATH=“.”
python setup.py develop
```

👆 [BACK to Table of Contents -->](#booksoutline)



### ⚒️3. Prepare the Environment

1. create environment and install pytorch

```shell
conda create --name picture python=3.8
conda activate picture
# install pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge
# Verify if pytorch is installed
import torch 
print(torch.cuda.is_available()) # If normal, return "True"
import torch    # If normal, remain silent
a = torch.Tensor([1.])    # If normal, remain silent
a.cuda()    # If normal, return"tensor([ 1.], device='cuda:0')"
from torch.backends import cudnn # If normal, remain silent
cudnn.is_acceptable(a.cuda())    # If normal, return "True"

```

2.install OpenPCDet

```shell
# install spconv
pip install spconv-cu111
# install requirements
pip install -r requirements.txt
# setup
python setup.py develop
```

3.install other packages

```shell
# install other packages
pip install torch_scatter
pip install nuscenes-devkit==1.0.5
pip install open3d

# install the Python package for evaluating the Waymo dataset
pip install waymo-open-dataset-tf-2-5-0==1.4.1

# pay attention to specific package versions.
pip install pandas==1.4.3
pip install matplotlib==3.6.2
pip install scikit-image==0.19.3
pip install async-lru==1.0.3

# install CUDA extensions
cd common_ops
pip install .
```

4.install MinkowskiEngine

```shell
# 安装MinkowskiEngine
pip install ninja
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

👆 [BACK to Table of Contents -->](#booksoutline)



### ⚒️4. Prepare the Seal Feature for the Entire Dataset Offline

1.Prepare the `coords` and `feats` inputs.

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_waymo_ssl_seal_generate_input.yaml
# or
python train.py --cfg_file cfgs/picture_models/picture_nuscenes_ssl_seal_generate_input.yaml
```

2.Utilize the MinkUNet (Res16UNet34C) pre-trained by Seal to generate the Seal features.

```shell
cd tools/
python prepare_seal_output.py
```

👆 [BACK to Table of Contents -->](#booksoutline)



### :rocket:5. Run the Code

We provide the configuration files in the paper and appendix in `tools/cfgs/picture_models/`. 

**Pre-training**

Waymo

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_waymo_ssl_seal_decoder_mask.yaml
```

nuScenes

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_nuscenes_ssl_seal_decoder_mask.yaml
```



**Fine-tuning**

3D Object Detection on Waymo:

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_waymo_detection.yaml --pretrained_model /path/of/pretrain/model.pth
```

3D Object Detection on nuScenes:

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_nuscenes_detection.yaml --pretrained_model /path/of/pretrain/model.pth
```



3D Semantic Segmentation:

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_nuscenes_segmentation.yaml --pretrained_model /path/of/pretrain/model.pth
```



Occupancy Prediction:

```shell
cd tools/
python train.py --cfg_file cfgs/picture_models/picture_nuscenes_occupancy.yaml --pretrained_model /path/of/pretrain/model.pth
```

👆 [BACK to Table of Contents -->](#booksoutline)
