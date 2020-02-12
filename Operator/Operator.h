#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Operator {
public:
    virtual Eigen::VectorXd apply(Eigen::VectorXd&) const =0;
};