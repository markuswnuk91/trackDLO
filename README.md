# trackDLO

Framework for Kinematic Model-based Tracking and Localization of Deformable Linear Objects

## Contents:

```
trackDLO
├──app/        # folder for demo applications
├──data        # folder for data management
    ├──darus_data_download          # folder DARUS integration for data management
        ├──data                     # folder for datasets
        ├──scripts                  # config and script files for data download
    ├──eval                         # folder for evaluation data
├──experimental/        # experimental scripts for trying out new things
├──eval/        # evaluation scripts for validation and evaluation of methodology
├──src/                 # source code
    ├──tracking         # tracking algorithms
        ├──spr/         # structure preserved registration         
├──tests/               # testing scripts for coding and algorithm development
├── README.md           # readme
└── SETUP.md            # install instructions and dependencies
```

## Installation

### Dependencies
Required libraries
```
# Open CV
sudo apt install libopencv-dev python3-opencv
```
Required Python packages
```
# for installation of Nerian API (libvisiontransfer)
pip3 Cython numpy wheel 
```

### Nerian API (libvisiontransfer)
Download the Nerian Repoitory from github
```
$ git clone git@github.com:nerian-vision/nerian-vision-software.git
```
Go in the folder and build the project
```
$ cd nerian-vision-software/
$ mkdir build
$ cd build
$ cmake ..
$ make
```
$ sudo make install