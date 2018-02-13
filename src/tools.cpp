#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse = VectorXd::Zero(estimations[0].size());
    for (int i = 0 ; i < estimations.size(); i++){
        VectorXd err = (estimations[i] - ground_truth[i]).array().square();
        rmse +=  err;
    }

    rmse /= estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;
}