#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


double normalize(double angle) {
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;

    return angle;
}

double NIS(VectorXd &obs, VectorXd &pred, MatrixXd &S) {
    VectorXd diff = obs - pred;
    double e = diff.transpose() * S * diff;
    return e;
}

/**
 *
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd::Identity(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

    use_laser_ = true;
    use_radar_ = true;

    R_Radar_ = MatrixXd::Zero(3, 3);
    R_Radar_(0, 0) = std_radr_ * std_radr_;
    R_Radar_(1, 1) = std_radphi_ * std_radphi_;
    R_Radar_(2, 2) = std_radrd_ * std_radrd_;

    R_Lidar_ = MatrixXd::Zero(2, 2);
    R_Lidar_(0, 0) = std_laspx_ * std_laspx_;
    R_Lidar_(1, 1) = std_laspy_ * std_laspy_;

    n_x_ = 5;
    n_aug_ = 7;
    n_aug_points_ = 1 + 2 * n_aug_;

    lambda_ = 3 - n_aug_;

    weights_ = VectorXd(n_aug_points_);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < weights_.size(); i++) {
        weights_(i) = 1. / (2 * (lambda_ + n_aug_));
    }

    std_a_ = .2;//0.5 * 3; //half of car's noise
    std_yawdd_ = .6;//M_PI/16.;

    is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    } else { //radar
        UpdateRadar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    double tol = 0.001;

    /********** step 1: generate sigma pts **********/

    MatrixXd augP = MatrixXd::Zero(P_.rows() + 2, P_.cols() + 2);
    MatrixXd Q = MatrixXd::Zero(2, 2);
    Q(0, 0) = std_a_ * std_a_;
    Q(1, 1) = std_yawdd_ * std_yawdd_;
    augP.topLeftCorner(P_.rows(), P_.cols()) = P_;
    //augP.block(P_.rows(), P_.cols(), 2, 2) = Q;
    augP(augP.rows() - 1, augP.cols() - 1) = std_yawdd_ * std_yawdd_;
    augP(augP.rows() - 2, augP.cols() - 2) = std_a_ * std_a_;

    Eigen::LLT<MatrixXd> lltOfPaug(augP);
    if (lltOfPaug.info() == Eigen::NumericalIssue) {
        cout << "numerical " << endl;
    }
    MatrixXd sqrP = augP.llt().matrixL();

//    sqrP *= sqrt(lambda_ + n_aug_);
    short r = sqrP.rows();
    short c = sqrP.cols();

    MatrixXd Aug = MatrixXd::Zero(n_aug_, 1 + 2 * n_aug_);

    VectorXd x_aug = VectorXd(x_.size() + 2);
    x_aug.head(x_.size()) = x_;
    x_aug(x_aug.size() - 2) = 0.;
    x_aug(x_aug.size() - 1) = 0.;

    Aug.col(0) = x_aug;

//    Xsig_pred_.block(0, 1, r, c) = sqrP.colwise() + x_aug;
//    Xsig_pred_.block(0, 1+ c, r, c) = (-sqrP).colwise() + x_aug;

    for (int i = 0; i < n_aug_; i++) {
        Aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * sqrP.col(i);
        Aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * sqrP.col(i);

        Aug(3, i + 1) = normalize(Aug(3, i + 1));
        Aug(3, i + 1 + n_aug_) = normalize(Aug(3, i + 1 + n_aug_));
    }

//    cout << "x aug before " << x_aug << endl;
//    cout << "P before " << P_ << endl;

    //step 2: predict sigma pts
    for (int i = 0; i < Aug.cols(); i++) {
        double v = Aug(2, i);
        double yaw = Aug(3, i);
        double yawRate = Aug(4, i);
        double aNoise = Aug(5, i);
        double omgNoise = Aug(6, i);

        if (fabs(yawRate) < tol) { //straight line
            Aug(0, i) += v * cos(yaw) * delta_t;
            Aug(1, i) += v * sin(yaw) * delta_t;
        } else {
            Aug(0, i) += (v / yawRate) * (sin(yaw + yawRate * delta_t) - sin(yaw));
            Aug(1, i) += (v / yawRate) * (-cos(yaw + yawRate * delta_t) + cos(yaw));
        }

        Aug(3, i) += yawRate * delta_t;

        //noise component
        Aug(0, i) += .5 * pow(delta_t, 2) * cos(yaw) * aNoise;
        Aug(1, i) += .5 * pow(delta_t, 2) * sin(yaw) * aNoise;
        Aug(2, i) += delta_t * aNoise;

        Aug(3, i) += .5 * pow(delta_t, 2) * omgNoise;
        Aug(3, i) = normalize(Aug(3, i));

        Aug(4, i) += delta_t * omgNoise;

    }


    //step 3: convert it back to mean and covariance
    Xsig_pred_ = Aug.block(0, 0, n_x_, Aug.cols());
    x_ = Xsig_pred_ * weights_;
    P_.fill(0.0);

    for (int i = 0; i < Xsig_pred_.cols(); i++) {
        VectorXd diff = Xsig_pred_.col(i) - x_;
        diff(3) = normalize(diff(3));
        P_ += weights_(i) * diff * diff.transpose();
    }

    cout << "x aug after: " << x_ << endl;
    cout << "P after:" << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {


    double std_v = 4.3; //TODO: justify


    //don't have to linearize. should be same as EKF proj
    if (!is_initialized_) {
        double x = meas_package.raw_measurements_(0), y = meas_package.raw_measurements_(1);
        double yaw = atan2(y, x);
        x_ << x, y, 10, 0, 0.5; //v, yaw, straight line
//        P_.fill(0.0);
//        P_(0, 0) = pow(std_laspx_, 2);
//        P_(1, 1) = pow(std_laspy_, 2);
//        P_(2, 2) = pow(std_v, 2);
//        P_(3, 3) = 1.;
//        P_(4, 4) = 1.;

        P_ = MatrixXd::Identity(5, 5);

        is_initialized_ = true;
        time_us_ = meas_package.timestamp_;

    } else if (use_laser_) {
        //prediction first.
        double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.;
        time_us_ = meas_package.timestamp_;
        Prediction(delta_t);

        return;

        MatrixXd H = MatrixXd::Zero(2, x_.size());
        H(0, 0) = 1;
        H(1, 1) = 1;

        MatrixXd S = H * P_ * H.transpose() + R_Lidar_;
        MatrixXd K = P_ * H.transpose() * S.inverse();
        MatrixXd I = MatrixXd::Identity(K.rows(), H.cols());

        VectorXd z = H * x_;
        VectorXd residual = meas_package.raw_measurements_ - z;
        x_ += K * residual;
        x_(3) = normalize(x_(3));
        MatrixXd gg = K * residual;
        P_ = (I - K * H) * P_;

        double nis = NIS(meas_package.raw_measurements_, z, S);
    }

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    VectorXd &m = meas_package.raw_measurements_;
    static int counter = 0;

    if (!is_initialized_) {
        is_initialized_ = true;
        double std_v = 4.3; //TODO: justify

        double r = m(0), phi = m(1);
        double x = r * cos(phi), y = r * sin(phi);

        x_ << x, y, 10, 0, .5; //v, yaw, straight line
        P_ = MatrixXd::Identity(5, 5);
//        P_.fill(0.0);
//        P_(0, 0) = pow(std_laspx_ * 1.5, 2) ;
//        P_(1, 1) = pow(std_laspy_ * 1.5, 2) ;
//        P_(2, 2) = pow(std_v, 2);
//        P_(3, 3) = 1.;
//        P_(4, 4) = 1.;

        time_us_ = meas_package.timestamp_;

    } else if (use_radar_) {
        cout << "doing radar" << endl;

        //prediction first.
        double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.;
        time_us_ = meas_package.timestamp_;
        Prediction(delta_t);

        return ;

        MatrixXd pred_mspace = MatrixXd::Zero(3, n_aug_points_);
        counter++;

        //step 1: predict sigma points
        for (int i = 0; i < n_aug_points_; i++) {
            VectorXd pt = Xsig_pred_.col(i);
            double x = pt(0);
            double y = pt(1);
            double v = pt(2);
            double yaw = pt(3);

            double r = sqrt(x * x + y * y);
            double orient = atan2(y, x);
            double rr = (x * cos(yaw) * v + y * sin(yaw) * v) / r;
            if (r == 0) {
                throw 5;
            }

            pred_mspace(0, i) = r;
            pred_mspace(1, i) = orient;
            pred_mspace(2, i) = rr;
        }

        //step 2: calculate innovation and innovation covariance
//        VectorXd z = pred_mspace * weights_;
        VectorXd z = VectorXd::Zero(3);
        for (int i = 0; i < n_aug_points_; i++){
            z += weights_(i) * pred_mspace.col(i);
        }

        z(1) = normalize(z(1));
        VectorXd residual = m - z;
        residual(1) = normalize(residual(1));


        MatrixXd S = MatrixXd::Zero(3, 3); //inno. cov

        for (int i = 0; i < pred_mspace.cols(); i++) {
            VectorXd diff = pred_mspace.col(i) - z;
            diff(1) = normalize(diff(1));
            S += weights_(i) * diff * diff.transpose();
        }

        S += R_Radar_;

        //step 3: calculate cross-correlation matrix T
        MatrixXd T = MatrixXd::Zero(n_x_, 3);
        for (int i = 0; i < pred_mspace.cols(); i++) {
            VectorXd d1 = Xsig_pred_.col(i) - x_;
            VectorXd d2 = pred_mspace.col(i) - z;

            d1(3) = normalize(d1(3));
            d2(1) = normalize(d2(1));

            T += weights_(i) * d1 * d2.transpose();
        }

        //step 4: calculate kalman gain
        MatrixXd K = T * S.inverse();

        //step 5: calculate new state cov and mean
        x_ += K * residual;
        P_ = P_ - K * S * K.transpose();

        double nis = NIS(m, z, S);
//        std::cout << "RADAR_NIS:" << nis << std::endl;
//        cout << "M" << m << endl;
//        cout << "z" << z << endl;
//        cout << "S" << S << endl;
//        cout << "P" << pred_mspace << endl;
    }

}

