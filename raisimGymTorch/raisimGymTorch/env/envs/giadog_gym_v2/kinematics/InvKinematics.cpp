#include "InvKinematics.hpp"

/*
    Calculates the leg's Inverse kinematicks parameters:
    The leg Domain 'D' (caps it in case of a breach) and the leg's radius.

    Arguments:
    ----------
        x: double  
            hip-to-foot distance in x-axis

        y: double  
            hip-to-foot distance in y-axis

        z: double  
            hip-to-foot distance in z-axis

    Returns:
    -------
        double
            leg's Domain D

        double
            leg's outer radius
*/
std::pair<double, double> get_IK_params(double x, double y, double z) 
{  
    double r_o, D, sqrt_component;

    sqrt_component = std::max(
        (double) 0, 
        std::pow(z, 2) + std::pow(y, 2) - std::pow(H_OFF, 2)
    );
    r_o = std::sqrt(sqrt_component) - V_OFF;
    D = std::clamp(
        (std::pow(r_o, 2) + std::pow(x, 2) - std::pow(SHANK_LEN, 2) - 
            std::pow(THIGH_LEN, 2)) / (2 * SHANK_LEN * THIGH_LEN),
        -1.0, 
        1.0
    );

    return {D, r_o};
}

/*
    Right Leg Inverse Kinematics Solver
    
    Arguments:
    ---------_
        x: double  
            hip-to-foot distance in x-axis

        y: double  
            hip-to-foot distance in y-axis

        z: double  
            hip-to-foot distance in z-axis
        
        D: double
            Leg domain
        
        r_o: double
            Radius of the leg

    Return:
    -------
        Eigen::Vector3d 
            Joint Angles required for desired position. 
            The order is: Hip, Thigh, Shank
            Or: (shoulder, elbow, wrist)
*/
Eigen::Vector3d right_leg_IK(
    double x, 
    double y, 
    double z, 
    double D, 
    double r_o
) {
    double wrist_angle, shoulder_angle, elbow_angle;
    double second_sqrt_component, q_o;

    wrist_angle    = std::atan2(-std::sqrt(1 - std::pow(D, 2)), D);
    shoulder_angle = - std::atan2(z, y) - std::atan2(r_o, - H_OFF);
    second_sqrt_component = std::max(
        (double) 0,
        std::pow(r_o, 2) + std::pow(x, 2) - 
            std::pow((SHANK_LEN * std::sin(wrist_angle)), 2)
    );
    q_o = std::sqrt(second_sqrt_component);
    elbow_angle = std::atan2(-x, r_o);
    elbow_angle -= std::atan2(SHANK_LEN * std::sin(wrist_angle), q_o);

    Eigen::Vector3d joint_angles(-shoulder_angle, elbow_angle, wrist_angle);

    return joint_angles;
}

/*
    Left Leg Inverse Kinematics Solver
    
    Arguments:
    ---------_
        x: double  
            hip-to-foot distance in x-axis

        y: double  
            hip-to-foot distance in y-axis

        z: double  
            hip-to-foot distance in z-axis
        
        D: double
            Leg domain
        
        r_o: double
            Radius of the leg

    Return:
    -------
        Eigen::Vector3d 
            Joint Angles required for desired position. 
            The order is: Hip, Thigh, Shank
            Or: (shoulder, elbow, wrist)
*/
Eigen::Vector3d left_leg_IK(
    double x, 
    double y, 
    double z, 
    double D, 
    double r_o
) {

    // Declare the variables
    double wrist_angle, shoulder_angle, elbow_angle;
    double second_sqrt_component, q_o;

    wrist_angle    = std::atan2(-std::sqrt(1 - std::pow(D, 2)), D);
    shoulder_angle = - std::atan2(z, y) - std::atan2(r_o, H_OFF);
    second_sqrt_component = std::max(
        (double) 0,
        std::pow(r_o, 2) + std::pow(x, 2) - 
            std::pow((SHANK_LEN * std::sin(wrist_angle)), 2)
    );
    q_o = std::sqrt(second_sqrt_component);
    elbow_angle = std::atan2(-x, r_o);
    elbow_angle -= std::atan2(SHANK_LEN * std::sin(wrist_angle), q_o);

    Eigen::Vector3d joint_angles(-shoulder_angle, elbow_angle, wrist_angle);

    return joint_angles;
}


Eigen::Vector3d solve_leg_IK(bool right_leg, Eigen::Vector3d r)
{
    auto [D, r_o] = get_IK_params(r(0), r(1), r(2));

    return right_leg ? 
        right_leg_IK(r(0), r(1), r(2), D, r_o) : 
        left_leg_IK(r(0), r(1), r(2), D, r_o);
}