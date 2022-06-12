/*
    This file is supposed to contain the enviroment variables:
    * The robot meassurements
    * Gait Parameters

*/
#if !defined(ENV_VARIABLES)
#define ENV_VARIABLES

#include <Eigen/Dense>

// Robot Meassurements
#define H_OFF     0.063
#define V_OFF     0.008
#define THIGH_LEN 0.11058
#define SHANK_LEN 0.1265
#define LEG_SPAN  0.2442
#define H         0.2 // Max foot height

// Gait Parameters
#define F0 4.0
#define FTG_TYPE "base"
#define SIM_SECONDS_PER_STEP 0.005
const Eigen::Vector3d HZ(0., 0., 1.);
const Eigen::Vector4d SIGMA_0(0., 3.142, 3.142, 0.);

#define CARTESIAN_DIR true
#define ANGULAR_DIR   true

#endif