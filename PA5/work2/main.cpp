//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
using namespace Eigen;

#include <iostream>
#include <sophus/so3.h>

using namespace std;

int main(int argc, char** argv) {
    // give a Essential
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    //compute E, t
    Matrix3d R;
    Vector3d t;

    //SVD and fix sigular values
    JacobiSVD<MatrixXd> svd(E, ComputeThinU | ComputeThinV);
    const Matrix3d& U = svd.matrixU();
    const Matrix3d& V = svd.matrixV();
    const Vector3d& singular_values = svd.singularValues();

    Matrix3d Sigma;
    Sigma << (singular_values[0] + singular_values[1])/2, 0, 0,
                        0, (singular_values[0] + singular_values[1])/2, 0,
                        0, 0, 0;
    /* U =
     * -0.0890846  -0.562354  -0.822084
     * 0.993441  -0.109576 -0.0326974
     * -0.0716928  -0.819605   0.568426
     * */

    /* V =
     *0.556696 -0.0369821   0.829892
     *0.0601827   0.998179 0.00411052
     *0.828533 -0.0476569  -0.557908
     **/

    /*
     * Sigma = 0.707107,0.707107,1.29353e-16
     **/

    /** four expressions
     *  1. Eigen::AngleAisd rotation_vector(M_PI/2, Eigen::Vector3d(0, 0, 1))  //r_z
     *
     *  2. Sophus::SO3 R_Z(rotation_vector.toRotationMatrix())
     *
     *  3. Sophus::SO3 R_Z(0, 0, M_PI/2)
     *
     *  4. Eigen::Quaterniond q(rotation_vector.toRotationMatrix()); Sophus::SO3 R_Z(q)
     * */
    // use AngleAxisd
/*
    Eigen::AngleAxisd R_Z(M_PI/2, Eigen::Vector3d(0, 0, 1));    //z rotation 90
    Eigen::AngleAxisd negative_R_Z(-1 * M_PI/2, Eigen::Vector3d(0, 0, 1)); //z rotation -90

    Matrix3d t_wedge1;
    Matrix3d t_wedge2;

    Matrix3d R1;
    Matrix3d R2;


   t_wedge1 = U * R_Z * Sigma * U.transpose();
    R1 = U * R_Z.matrix().transpose() * V.transpose();

    t_wedge2 = U * negative_R_Z * Sigma * U.transpose();
    R2 = U * negative_R_Z.matrix().transpose() * V.transpose();
*/
    /*
     * the relationship between T and the multiple of the answers in the book
     * */
 /*
    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;
*/
    //use Sophus::SO3
    Sophus::SO3 R_Z(0, 0, M_PI/2);
    Sophus::SO3 negative_R_Z(0, 0, -M_PI/2);

    Matrix3d t_wedge1;
    Matrix3d t_wedge2;

    Matrix3d R1;
    Matrix3d R2;

    t_wedge1 = U * R_Z.matrix() * Sigma * U.transpose();
    R1 = U * R_Z.matrix().transpose() * V.transpose();

    t_wedge2 = U * negative_R_Z.matrix() * Sigma * U.transpose();
    R2 = U * negative_R_Z.matrix().transpose() * V.transpose();

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}