//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.hpp"
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

namespace raisim {


class RaisimGymEnv {

 public:
  explicit RaisimGymEnv (std::string resourceDir, const Yaml::Node& cfg, int port=8080) :
      resourceDir_(std::move(resourceDir)), cfg_(cfg), port_(port) { }

  virtual ~RaisimGymEnv() { if(server_) server_->killServer(); };

  /////// implement these methods /////////
  virtual void init() = 0;
  virtual void reset() = 0;
  virtual void observe(Eigen::Ref<EigenVec> ob) = 0;
  virtual float step(const Eigen::Ref<EigenVec>& action) = 0;
  virtual bool isTerminalState(float& terminalReward) = 0;
  ////////////////////////////////////////

  /////// optional methods ///////
  virtual void getBaseEulerAngles(Eigen::Ref<EigenVec> ea) = 0;
  virtual void curriculumUpdate() {};
  virtual void hills(double frequency, double amplitude, double roughness) = 0;
  virtual void stairs(double width, double height) = 0;
  virtual void cellularSteps(double frequency, double amplitude) {};
  virtual void steps(double width, double height) {};
  virtual void slope(double slope, double roughness) {};
  virtual void demoTerrain(int n_objects, double radius) {};
  virtual double getTraversability(void) = 0;
  virtual double getSpeed(void) {return 0.0;};
  // Quadruped statistics
  virtual double get_max_torque(void){return 0.0;};
  virtual double get_power(void){return 0.0;};
  virtual double get_froude(void){return 0.0;};
  virtual double get_proj_speed(void){return 0.0;};
  virtual void close() {};
  virtual void setSeed(int seed) {};
  virtual void setCommand(double direction_angle, double turning_direction, bool stop) {};
  ////////////////////////////////

  void setSimulationTimeStep(double dt) { simulation_dt_ = dt; world_->setTimeStep(dt); }
  void setControlTimeStep(double dt) { control_dt_ = dt; }
  int getObDim() { return obDim_; }
  std::map<std::string, std::vector<int>> getObIndexDict() { return ob_idx_dict_; }
  int getActionDim() { return actionDim_; }
  double getControlTimeStep() { return control_dt_; }
  double getSimulationTimeStep() { return simulation_dt_; }
  raisim::World* getWorld() { return world_.get(); }
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  raisim::Reward& getRewards() { return rewards_; }

 protected:
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::string resourceDir_;
  Yaml::Node cfg_;
  int obDim_=0, actionDim_=0;
  std::map<std::string, std::vector<int>> ob_idx_dict_;
  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Reward rewards_;
  int port_;
};
}

#endif //SRC_RAISIMGYMENV_HPP