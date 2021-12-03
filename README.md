# TSGL-EEGConv

This is the Repository of TSGL-EEGNet. TSGL-EEGNet is a kind of Convolutional Network for pre-processed EEG signals to predict classes. TSGL-EEGNet is based on EEGNet which you can find it [here](https://github.com/vlawhern/arl-eegmodels). 

## Dataset

- [BCI Competition IV 2a](http://www.bbci.de/competition/iv/#dataset2a)

You need to do some pre-processing works before using it. Maybe you need MATLAB and EEGlab tools to read these datas and transform them to *.mat file format.

## Features

### Framework for EEG

- Cross Validation Framework
- Grid Search Framework
- Cropped Training with CV and GS
- Visualization Framework

### Implemented Models

- EEGNet [[1]](http://stacks.iop.org/1741-2552/15/i=5/a=056013)
- TSGL-EEGNet
- DeepConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ~~Multi-branch 3D CNN [[3]](https://ieeexplore.ieee.org/document/8820089)~~ Not supported yet.
- [FBCSP](https://github.com/TNTLFreiburg/fbcsp)

### Upgrade Log

- v1.1
  - use a new code construction to do cropped training
  - use generator to train on a large dataset
  - rename core.train to core.training, to be different from .train
  - add maxmin normalization
  - separate code test and model test
  - fix some bugs
- known issues
  - cv testing and model ensemble (stacking) are not adapted
- next
  - use generator to load a large dataset

## Usage

### Requirement

- Python >= 3.6 && <= 3.8
- Python >= 3.7 (when your GPU is newer than Nvidia 30xx)
- tensorflow-gpu >= 2.0.0 && <= 2.3.0
- tensorflow >= 2.4.1 (pip) (when your GPU is newer than Nvidia 30xx)
- scikit-learn >= 0.21.3
- scipy >= 1.3.1
- numpy >= 1.17.3
- tensorflow-addons (pip)
- hdf5 >= 1.10.4
- h5py >= 2.9.0
- matplotlib >= 3.1.1 && <=3.3.4 (optional)
- pydot >= 1.4.1 (optional)
- graphviz >= 2.38 (optional)
- mne >= 0.20.7 (pip) (optional)
- braindecode == 0.2.0 (pip) (fbcsp) (optional)

It is recommended to use conda environment.
tensorflow-addons [Requirement](https://github.com/tensorflow/addons#python-op-compatibility-matrix)
Optional packages are for visualization

### Quick Start

```shell
# training models
python train.py

# testing single model
python model_test.py

# testing CV models (not adapted)
# python model_cv.py

# ensemble models (not adapted)
# python model_ensemble.py

# stacking models (not adapted)
# python model_stacking.py

# visualization
python vis.py
```

### Code Structure

TODO

# Paper Citation

If you use the EEGNet model in your research and found it helpful, please cite the following paper:

    @article{Lawhern2018,
        author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
        title={EEGNet: a Compact Convolutional Neural Network for EEG-based Brain–computer Interfaces},
        journal={Journal of Neural Engineering},
        volume={15},
        number={5},
        pages={056013},
        url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
        year={2018}
    }
    
If you use the TSGL-EEGNet model in your research and find this Repository is helpful, please cite the following paper:

    @article{Deng2021,
        author={Deng, Xin and Zhang, Boxian and Yu, Nian and Liu, Ke and Sun, Kaiwei},
        journal={IEEE Access},
        title={Advanced TSGL-EEGNet for Motor Imagery EEG-Based Brain-Computer Interfaces},
        year={2021},
        volume={9},
        number={},
        pages={25118-25130},
        doi={10.1109/ACCESS.2021.3056088}
    }

# License

MIT License
