/*

*/
#include <vector>
#include <Eigen/Dense>

#include "Gait.hpp"
#include "../__env__.hpp"
#include "InvKinematics.hpp"
#include "TransfMatrices.hpp"

/*

*/
std::tuple<Eigen::VectorXd, Eigen::Vector4d, Eigen::VectorXd> calculate_joint_angles(
    Eigen::Vector2d command,
    Eigen::VectorXd NN_output,
    Eigen::Vector3d base_rpy,
    double time_step
);
