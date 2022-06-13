/*
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Inverse kinematics class for the giadog robot. (Spot mini mini // Open Quadruped)
    
    References:
    -----------
        * Muhammed Arif Sen, Veli Bakircioglu, Mete Kalyoncu. (Sep, 2017). 
        Inverse Kinematic Analysis Of A Quadruped Robot  
        https://www.researchgate.net/publication/320307716_Inverse_Kinematic_Analysis_Of_A_Quadruped_Robot

        * Some of the code was taken from the sopt_mini_mini implementation 
        of the same paper.
        https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py

*/
#pragma once

#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>

#include "../__env__.hpp"


/*
    Calculates the leg's inverse kinematics.
    (joint angles from xyz coordinates).

    Arguments:
    ---------_
        right_leg: bool 
            ('l' or 'r') 
            If true, the right leg is solved, otherwise the left leg is solved.
        
        r: Eigen::Vector3d
            Objective foot position in the H_i frame.
            (x,y,z) hip-to-foot distances in each dimension
            The 

    Return:
    -------
        Eigen::Vector3d
            Leg joint angles to reach the objective foot position r. In the 
            order:(Hip, Shoulder, Wrist). The joint angles are expresed in 
            radians.
*/
Eigen::Vector3d solve_leg_IK(bool right_leg, Eigen::Vector3d r);
