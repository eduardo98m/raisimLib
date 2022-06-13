/* 
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    The file contains the base gai class for the foot trajectory generation.

*/
#pragma once

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
        BaseGait(const Eigen::Vector4d sigma_0, double f0, const Eigen::Vector3d hz)
        {
            this->sigma_0 = sigma_0;
            this->f0 = f0;
            this->hz = hz;
        }
        
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
        ) {
            Eigen::Vector3d position;
            double sigma_i, k, h;
            bool condition_1, condition_2;

            sigma_i = std::fmod(sigma_i_0 + t * (this->f0 + f_i), (2 * M_PI));
            k       = 2 * (sigma_i - M_PI) / M_PI;
            h       = 0.8 * H;

            condition_1 = (k <= 1 && k >= 0);
            condition_2 = (k >= 1 && k <= 2);

            position += h * (-2 * k * k * k + 3 * k * k)  * this->hz * condition_1;
            position += h * (2 * k * k * k - 9 * k * k + 12 * k - 4) * 
                this->hz * condition_2;

            // See that if no condition is meet the position is the vector 0
            return {position, sigma_i}; 
        }

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
        compute_foot_trajectories(float t, Eigen::Vector4d frequencies)
        {
            Eigen::Vector4d FTG_frequencies;
            Eigen::MatrixXd FTG_phases(4, 2);
            Eigen::MatrixXd target_foot_positions(4, 3);

            for (int i = 0; i < 4; i++) {
                auto [r, sigma_i] = this->FTG(this->sigma_0[i], t, frequencies[i]);
                FTG_frequencies[i] = this->f0 + frequencies[i];
                FTG_phases(i, 0)   = std::sin(sigma_i);
                FTG_phases(i, 1)   = std::cos(sigma_i);
                target_foot_positions.row(i) = r;
            }

            return {target_foot_positions, FTG_frequencies, FTG_phases};
        }
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
) {
    Eigen::MatrixXd delta(4, 3);

    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d position_delta(0, 0, 0);
        
        // Position delta
        position_delta(0) = 1.7 * cos(command_dir) * delta_t * ftg_sine_phases(i) * 
            ftg_freqs(i) * LEG_SPAN;
        position_delta(1) = 1.02 * sin(command_dir) * delta_t * ftg_sine_phases(i) *
            ftg_freqs(i) * LEG_SPAN;

        // Rotation delta (Look Mom no branching!!)
        Eigen::Vector3d rotation_delta(0, 0, 0);
        
        double theta = M_PI/4;
        double phi_arc = (i == 0) * -theta + (i == 1) * -(M_PI - theta) + 
            (i == 2) *  theta + (i == 3) * (M_PI - theta);
        
        rotation_delta(0) = 0.68 * -cos(phi_arc) * delta_t * ftg_sine_phases(i) * 
            turn_dir * ftg_freqs(i) * LEG_SPAN;
        rotation_delta(1) = 0.68 * -sin(phi_arc) * delta_t * ftg_sine_phases(i) * 
            turn_dir * ftg_freqs(i) * LEG_SPAN;
        
        delta.row(i) =  (position_delta * add_cartesian_delta + \
                         rotation_delta * add_cartesian_delta);
        }

    return delta;
}

