/* 
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    The file contains the base gai class for the foot trajectory generation.

*/
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include "../__env__.hpp"

/*
    Class that represent the simple bezier FTG, used in:
    
    https://arxiv.org/pdf/2010.11251.pdf
*/
class BaseGait {
    
    public:
        double f0;
        Eigen::Vector3d hz;
        Eigen::Vector4d sigma_0;

        BaseGait(void) { }
        BaseGait(const Eigen::Vector4d sigma_0, double f0, const Eigen::Vector3d hz);
        
        /*
            Generates a vector in R^3 representing the desired foot position
            (end efector) in the H_i frame corresponding to the robots i-th 
            leg horizontal frame below its hip.
            
            Arguments:
            ----------
                sigma_i_0 : float 
                    Contact phase.

                t  : float 
                    Timestep.

                f_i: float 
                    i-th leg frequency offset (from NN policy).
        */
        std::pair<Eigen::Vector3d, double> FTG(
            double sigma_i_0,
            double t,
            double f_i
        );

        /*
            Compute the foot trajectories for the given frequencies, for all
            the four legs for the current time t.

            Arguments:
            ----------
                t : float
                    Current time.

                frequencies : Eigen::Vector4d
                    Vector of the four frequencies offsets of the four legs.
        */
        std::tuple<Eigen::MatrixXd, Eigen::Vector4d, Eigen::MatrixXd> 
        compute_foot_trajectories(float t, Eigen::Vector4d frequencies);
};

/*
    Function to add directionality to gaits.

    Is based on an heuristic similar to the one used by Raibert.

    Notes:
    The constants might be changed to be more general.

    Arguments:
    ----------
        delta_t: double
            Time step.
        
        ftg_freqs: Eigen::Vector4d
            The frequencies of the four legs.
        
        ftg_sine_phases: Eigen::Vector4d
            The phases of the four legs.

        command_dir: double
            The desired direction of the robot. (It is an angle in radians)
        
        turn_dir: int
            The direction of the turn. 1 for clockwise, -1 for 
            counterclockwise, 0 for no turn.
        
        add_cartesian_delta: bool
            Whether to add the cartesian delta (position delta).
        
        add_angular_delta: bool
            Whether to add the angular delta (rotation delta).
*/
Eigen::MatrixXd direction_deltas(
    double delta_t, 
    Eigen::Vector4d ftg_freqs, 
    Eigen::Vector4d ftg_sine_phases, 
    double command_dir, 
    int turn_dir,
    bool add_cartesian_delta,
    bool add_angular_delta
);

