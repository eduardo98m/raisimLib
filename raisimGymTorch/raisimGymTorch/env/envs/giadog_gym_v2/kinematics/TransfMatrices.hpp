/*
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Functions to generate the transformation matrices from the roboto hips to 
    the horizontal frames.

*/

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "../__env__.hpp"

/*
    Calculates a rotation matrix from the euler angles.

    Arguments:
    ----------
        roll: double
            roll angle
        
        pitch: double
            pitch angle
        
        yaw: double
            yaw angle
    
    Returns:
    -------
        rotationMatrix: Eigen::Matrix3d
            Rotation matrix.
*/
Eigen::Matrix3d rotation_matrix_from_euler(double roll, double pitch, double yaw);

/*
    Returns the transformation matrices from the hip to the leg base.

    Arguments:
    ---------
        base_rpy: numpy.array, shape(3,) 
            The hip's (and robot base) euler angles. (roll, pitch, yaw)

    Returns:
    --------
        Eigen::MatrixXd, shape (4,4). 
            A list containing the transformation matrices from the hip to 
            the leg base, for each of the robots legs.
        The order of the matrices is: LF, RF, LB, RB.
*/
std::vector<Eigen::Matrix4d> transf_matrices(Eigen::Vector3d base_rpy);