# TODOs

## Plotting
- [x] Setup callback: set assignment of variables to input arguments outside of base class
- [ ] Make nice looking design for DLO plots for thesis
- [x] Make Plot for discretization criteria: Continous DLO and corresponding discrete DLO
- [x] Plot for model generation / BDLO Topologies
- [x] Plot for Localization (Raw point coud --> random sample --> SOM --> L1 --> MinSpanTree)

## Modelling
- [x] Code discrete DLO class
- [x] discrete DLO class
- [x] topology model for BDLO

## Reconstruction
- [x] reconstuction for continous DLO representation
- [x] reconstuction for discrete DLO representation
- [x] reconstruction for BDLO representation

## Simulation
- [x] DLO model generation
- [x] BDLO model generation

## Tracking
- [x] implement cpd
- [x] implement spr
- [x] implement bspr
- [x] implement jacobian based registration methods
- [ ] implement branched based correspondance estimation 

## Localization
- [x] transfer Self Organizing Map algorithm
- [x] transfer L1-Median algorithm
- [x] transfer MLLE algorithm
- [x] implement filters
- [x] transfer Minimum Spanning-Tree algorithms
- [x] branch wise correspondance estimation
- [x] heuristic for restarting downsampling if less or more branches than expected are foudn (less branches --> double the number of input points, more branches --> half the number of seedPoints)
- [ ] localization based on reconstruction for BDLO

## Validation
- [x] determine Validation scenarios
- [ ] implement evaluation 
    - [ ] comparison of tracking performance of different algorithms (on static configurations and dynamic manipulation sequence)
    - [ ] evaluation of accuracy and robustness of initial localization (on static configurations)
    - [ ] comparison of tracking performance with and without inital localization (on static configurations)
    - [ ] evaluation of overall pipeline (localizaiton + tracking) for robotic manipulation (grasping error)

- [ ] determine performance metrics
    - [x] Accuracy
        - overall tracking error (coverage of point cloud by model)
        - length error (error betwwen known length and measured length)
        - uniformity error (error betwwen points)
        - label error (error between label in point cloud and label on model)
        - translational grasping error (translational error between estimated grasping position and measured grasping position)
        - angular grasping error (angular error between estimated grasping position and measured grasping position)
    - [ ] Runtime   

## Data Acquisiton
- [x] acquire data sets