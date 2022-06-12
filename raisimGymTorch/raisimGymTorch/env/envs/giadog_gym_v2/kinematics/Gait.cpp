#include "Gait.hpp"

BaseGait::BaseGait(const Eigen::Vector4d sigma_0, double f0, const Eigen::Vector3d hz)
{
    this->sigma_0 = sigma_0;
    this->f0 = f0;
    this->hz = hz;
}

std::pair<Eigen::Vector3d, double> BaseGait::FTG(
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

std::tuple<Eigen::MatrixXd, Eigen::Vector4d, Eigen::MatrixXd> 
BaseGait::compute_foot_trajectories(float t, Eigen::Vector4d frequencies)
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