/*

*/
#pragma once

#include <vector>
#include <Eigen/Dense>

#include "Gait.hpp"
#include "../__env__.hpp"
#include "InvKinematics.hpp"
#include "TransfMatrices.hpp"

class FTG_Handler
{
    public:
        double dt; 
        BaseGait gait;

        FTG_Handler(std::string FTG_type) 
        {    
            if (FTG_type == "base")
            {
                this->gait = BaseGait(SIGMA_0, F0, HZ); 
                this->dt = SIM_SECONDS_PER_STEP;   
            }
            else
            {
                throw std::runtime_error(
                    "The specified FTG: " + FTG_type + " is not supported."
                );
            }
        };
        
        /*
            Generates the foot trajectories for the given theta and t.
        */
        std::tuple<Eigen::MatrixXd, Eigen::Vector4d, Eigen::MatrixXd> gen_trajectories(
            Eigen::Vector2d command, 
            Eigen::VectorXd theta, 
            double t,
            bool cartesian_directionality,
            bool angular_directionality
        ) {
            Eigen::MatrixXd xyz_residual;
            xyz_residual << 
                theta.coeff(0), theta.coeff(1), theta.coeff(2),
                theta.coeff(3), theta.coeff(4), theta.coeff(5),
                theta.coeff(6), theta.coeff(7), theta.coeff(8),
                theta.coeff(9), theta.coeff(10), theta.coeff(11)   
            ;
            Eigen::Vector4d frequencies = theta.segment(12, 4);

            double command_dir = command[0];
            double turn_dir = command[1];
            auto [foot_target_pos, ftg_freqs, ftg_phases] = 
                this->gait.compute_foot_trajectories(t, frequencies);
            
            Eigen::MatrixXd dir_deltas = direction_deltas(
                this->dt,
                ftg_freqs,
                ftg_phases.row(0),
                command_dir,
                turn_dir,
                cartesian_directionality,
                angular_directionality
            );
            foot_target_pos +=  xyz_residual + dir_deltas;

            return std::make_tuple(foot_target_pos, ftg_freqs, ftg_phases);
        }
};

FTG_Handler foot_trajectory_geneator(FTG_TYPE);

/*

*/
std::tuple<Eigen::VectorXd, Eigen::Vector4d, Eigen::VectorXd> calculate_joint_angles(
    Eigen::Vector2d command,
    Eigen::VectorXd NN_output,
    Eigen::Vector3d base_rpy,
    double time_step
) {  
    //Initialize the joint angle vector
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(12);
    // Calculate the transformation matrices for the base and the legs.
    std::vector<Eigen::Matrix4d> T_matrices = transf_matrices(base_rpy);

    // For each leg, calculate the foot position and the joints angles.
    auto trajectory = foot_trajectory_geneator.gen_trajectories(
        command, 
        NN_output, 
        time_step, 
        CARTESIAN_DIR,
        ANGULAR_DIR
    );
    auto [foot_positions, ftg_freqs, ftg_phases] = trajectory;

    for (int i = 0; i < 4; i++)
    {   
        Eigen::Matrix4d T_i = T_matrices[i];

        Eigen::Vector3d foot_position_leg_frame;

        foot_position_leg_frame = T_i.topLeftCorner(3,3) * foot_positions(i) +
            T_i.topRightCorner(3,1);

        bool right_leg = (i == 1 || i == 3);
        // Calculate the joint angles.
        Eigen::Vector3d joint_angles_leg;
        joint_angles_leg = solve_leg_IK(right_leg, foot_position_leg_frame);

        // Add the joint angles to the joint angle vector.
        joint_angles(i*4)   = joint_angles_leg.coeff(0);
        joint_angles(i*4+1) = joint_angles_leg.coeff(1);
        joint_angles(i*4+2) = joint_angles_leg.coeff(2);

    }
    return std::make_tuple(joint_angles, ftg_freqs, ftg_phases);

}