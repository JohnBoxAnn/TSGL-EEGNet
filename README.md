# TSGL-EEGConv

This is the Repository of TSGL-EEGNet. TSGL-EEGNet is a kind of Convolutional Network for pre-processed EEG signals to predict classes. TSGL-EEGNet is based on EEGNet which you can find it [here](https://github.com/vlawhern/arl-eegmodels). 

## Dataset

- [BCI Competition IV 2a](http://www.bbci.de/competition/iv/#dataset2a)

You need to do some pre-processing works before using it. Maybe you need MATLAB and EEGlab tools to read these datas and transform them to *.mat file format.

## Features

# Framework for EEG

- Cross Validation Framework
- Grid Search Framework
- Cropped Training with CV and GS
- Visualization Framework

# Implemented Models

- EEGNet [1](http://stacks.iop.org/1741-2552/15/i=5/a=056013)
- TSGL-EEGNet
- DeepConvNet [2](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [2](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- Multi-branch 3D CNN [3](https://ieeexplore.ieee.org/document/8820089)
- [FBCSP](https://github.com/TNTLFreiburg/fbcsp)

## Usage

### Requirement

- Python >= 3.6
- tensorflow-gpu == 2.1.0
- scikit-learn >= 0.21.3
- scipy >= 1.3.1
- numpy >= 1.17.3
- pydot >= 1.4.1
- hdf5 >= 1.10.4
- h5py >= 2.9.0
- matplotlib >= 3.1.1
- graphviz >= 2.38
- mne >= 0.20.7 (pip)

It is recommended to use conda environment.

### Coding

    ```python
    import BIEEGConv as bieeg
    # TODO: finish the doc
    ```

# Paper Citation

If you use the EEGNet model in your research and found it helpful, please cite the following paper:

    @article{Lawhern2018,
        author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
        title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
        journal={Journal of Neural Engineering},
        volume={15},
        number={5},
        pages={056013},
        url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
        year={2018}
    }

# License

MIT License