//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <set>
#include <stdlib.h>
#include <Eigen/Dense>

#include "kinematics/FTG.hpp"
#include "../../RaisimGymEnv.hpp"

namespace raisim 
{

    class ENVIRONMENT : public RaisimGymEnv 
    {

        private:
            int gcDim_, gvDim_, nJoints_;
            bool visualizable_ = false;
            raisim::ArticulatedSystem* anymal_;
            Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
            double terminalRewardCoeff_ = -10., timeStep_ = 0.0;
            Eigen::VectorXd actionMean_, actionStd_, obDouble_;
            Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyOrientation_;
            std::set<size_t> footIndices_;

            /// these variables are not in use. They are placed to show you how to create a random number sampler.
            std::normal_distribution<double> normDist_;
            thread_local static std::mt19937 gen_;

        public:

            explicit ENVIRONMENT(
                const std::string& resourceDir, 
                const Yaml::Node& cfg, 
                bool visualizable
            ) : RaisimGymEnv(resourceDir, cfg), 
                visualizable_(visualizable), 
                normDist_(0, 1) 
            {
                /// create world
                this->world_ = std::make_unique<raisim::World>();

                /// add objects
                // anymal_ = this->world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
                this->anymal_ = this->world_->addArticulatedSystem(
                    resourceDir_ + "/giadog/mini_ros/urdf/spot.urdf"
                );
                this->anymal_->setName("anymal");
                this->anymal_->setControlMode(
                    raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE
                );
                this->world_->addGround();

                /// get robot data
                this->gcDim_ = this->anymal_->getGeneralizedCoordinateDim();
                this->gvDim_ = this->anymal_->getDOF();
                this->nJoints_ = this->gvDim_ - 6;

                /// initialize containers
                this->gc_.setZero(this->gcDim_); 
                this->gc_init_.setZero(this->gcDim_);
                this->gv_.setZero(this->gvDim_); 
                this->gv_init_.setZero(this->gvDim_);
                this->pTarget_.setZero(this->gcDim_); 
                this->vTarget_.setZero(this->gvDim_); 
                this->pTarget12_.setZero(this->nJoints_);

                /// this is nominal configuration of the quadruped
                this->gc_init_ << 
                    0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, 
                    -0.03, 0.4, -0.8, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8
                ;
                /// Connfiguracion: [x,y,z, qr, qi, qj, qk, 12 joints angles]
                /// set pd gains
                Eigen::VectorXd jointPgain(this->gvDim_), jointDgain(this->gvDim_);
                jointPgain.setZero(); 
                jointPgain.tail(this->nJoints_).setConstant(50.0);
                jointDgain.setZero(); 
                jointDgain.tail(this->nJoints_).setConstant(0.2);
                this->anymal_->setPdGains(jointPgain, jointDgain);
                this->anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(this->gvDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_ = 34;
                actionDim_ = 16; 
                this->actionMean_.setZero(actionDim_); 
                this->actionStd_.setZero(actionDim_);
                this->obDouble_.setZero(obDim_);

                /// action scaling
                //this->actionMean_ = this->gc_init_.tail(this->nJoints_);
                this->actionMean_ = Eigen::VectorXd::Zero(16);
                //this->actionStd_.setConstant(0.3);
                this->actionStd_.tail(4).setConstant(0.1);
                this->actionStd_.head(12).setConstant(0.03);

                /// Reward coefficients
                rewards_.initializeFromConfigurationFile (cfg["reward"]);

                /// indices of links that should not make contact with ground
                this->footIndices_.insert(this->anymal_->getBodyIdx("back_right_lower_leg"));
                this->footIndices_.insert(this->anymal_->getBodyIdx("front_right_lower_leg"));
                this->footIndices_.insert(this->anymal_->getBodyIdx("back_left_lower_leg"));
                this->footIndices_.insert(this->anymal_->getBodyIdx("front_left_lower_leg"));

                /// visualize if it is the first environment
                if (this->visualizable_) 
                {
                    server_ = std::make_unique<raisim::RaisimServer>(this->world_.get());
                    server_->launchServer();
                    server_->focusOn(this->anymal_);
                }
            }

            void init() final { }

            void reset() final 
            {
                this->anymal_->setState(this->gc_init_, this->gv_init_);
                updateObservation();
                this->timeStep_ = 0;
            }

            float step(const Eigen::Ref<EigenVec>& action) final 
            {
                /// action scaling
                //this->pTarget12_ = action.cast<double>();
                Eigen::Vector2d comand(0,0);
                auto [pTarget12_aux, ftg_freqs, ftg_phases] = calculate_joint_angles(
                    comand,
                    action.cast<double>(),
                    this->bodyOrientation_,
                    this->timeStep_
                );
                this->pTarget12_ = pTarget12_aux;
                this->pTarget12_ = this->pTarget12_.cwiseProduct(this->actionStd_);
                this->pTarget12_ += this->actionMean_;
                this->pTarget_.tail(this->nJoints_) = this->pTarget12_;

                this->anymal_->setPdTarget(this->pTarget_, this->vTarget_);

                for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++)
                {
                    if(server_) server_->lockVisualizationServerMutex();
                    this->world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                this->timeStep_ += control_dt_;

                updateObservation();

                rewards_.record("torque", this->anymal_->getGeneralizedForce().squaredNorm());
                rewards_.record("forwardVel", std::min(4.0, this->bodyLinearVel_[0]));

                return rewards_.sum();
            }

            void updateObservation() 
            {
                this->anymal_->getState(this->gc_, this->gv_);
                raisim::Vec<4> quat;
                raisim::Mat<3,3> rot;
                quat[0] = this->gc_[3]; 
                quat[1] = this->gc_[4];
                quat[2] = this->gc_[5]; 
                quat[3] = this->gc_[6];
                raisim::quatToRotMat(quat, rot);
                this->bodyLinearVel_ = rot.e().transpose() * this->gv_.segment(0, 3);
                this->bodyAngularVel_ = rot.e().transpose() * this->gv_.segment(3, 3);

                // get the orientation of the body
                this->bodyOrientation_ = rot.e().row(2).transpose();

                this->obDouble_ << 
                    this->gc_[2], /// body height
                    this->bodyOrientation_, /// body orientation
                    this->gc_.tail(12), /// joint angles
                    this->bodyLinearVel_, 
                    this->bodyAngularVel_, /// body linear&angular velocity
                    this->gv_.tail(12) /// joint velocity
                ;
            }

            void observe(Eigen::Ref<EigenVec> ob) final 
            {
                /// convert it to float
                ob = this->obDouble_.cast<float>();
            }

            bool isTerminalState(float& terminalReward) final 
            {
                terminalReward = float(this->terminalRewardCoeff_);
                inline std::set<size_t>::iterator contact_find;

                /// if the contact body is not feet
                for(auto& contact: this->anymal_->getContacts())
                {
                    contact_find = this->footIndices_.find(contact.getlocalBodyIndex());
                    if (contact_find == this->footIndices_.end())
                    return true;
                }

                terminalReward = 0.f;
                return false;
            }

            void curriculumUpdate() { };
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

