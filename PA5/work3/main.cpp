//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include "sophus/se3.h"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>>    VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector2d>>    VecVector2d;
typedef Matrix<double, 6, 1>    Vector6d;

const string p3d_file = "../p3d.txt";
const string p2d_file = "../p2d.txt";

int main(int argc, char** argv){
    VecVector3d p3d;
    VecVector2d p2d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;            //camera pose

    ifstream fin_p3d(p3d_file);
    if(!fin_p3d){
        cerr << "can not find p3d_file" << endl;
        return 1;
    }

    ifstream fin_p2d(p2d_file);
    if(!fin_p2d){
        cerr << "can not find p2d_file" << endl;
        return 1;
    }

    string line1;
    while(getline(fin_p3d, line1)){
        std::stringstream stringstream1(line1);
        double p3d_data[3] = {0};
        for(auto& p3d_d : p3d_data){
            stringstream1 >> p3d_d;
        }
        Eigen::Vector3d p3d_d(p3d_data[0], p3d_data[1], p3d_data[2]);
        p3d.push_back(p3d_d);
    }
/*
    for(auto& p3 : p3d){
        cout << p3.matrix().transpose() << endl;
    }
*/
    string line2;
    while(getline(fin_p2d, line2)){
        std::stringstream stringstream2(line2);
        double p2d_data[2] = {0};
        for(auto& p2d_d : p2d_data){
            stringstream2 >> p2d_d;
        }
        Eigen::Vector2d p2d_d(p2d_data[0], p2d_data[1]);
        p2d.push_back(p2d_d);
    }
/*
    for(auto& p2 : p2d){
        cout << p2.matrix().transpose() << endl;
    }
*/
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    unsigned long nPoints = p3d.size();
    cout << "points:" << nPoints << endl;

    Sophus::SE3 T_esti; //estimated pose

    for(int iter = 0; iter < iterations; iter++){
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d g = Vector6d::Zero();
        cost = 0;
        for(int i = 0; i < nPoints; i++){
            //compute cost for p3d[I] and p2d[I]
            double Xw = p3d[i][0], Yw = p3d[i][1], Zw = p3d[i][2];         // world coordinate
            double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

            Vector3d P(Xw, Yw, Zw);
            Vector3d Pc = T_esti * P;                                      //camera coordiante

            double X = Pc[0], Y = Pc[1], Z = Pc[2];
            Eigen::Vector2d e;
            e(0, 0) = p2d[i][0] - fx*X/Z - cx;
            e(1, 0) = p2d[i][1] - fy*Y/Z - cy;

            cost += e.matrix().transpose() * e;
            //compute jacobian
            Matrix<double, 2, 6> J;
            double Z_2 = pow(Z, 2), X_2 = pow(X, 2), Y_2 = pow(Y, 2);

            J(0, 0) = -1 * fx/Z;
            J(0, 1) = 0;
            J(0, 2) = fx*X/Z_2;
            J(0, 3) = fx*X*Y/Z_2;
            J(0, 4) = -fx - fx*X_2/Z_2;
            J(0, 5) = fx*Y/Z;

            J(1, 0) = 0;
            J(1, 1) = -fy/Z;
            J(1, 2) = fy*Y/Z_2;
            J(1, 3) = fy + fy*Y_2/Z_2;
            J(1, 4) = -fy*Y*X/Z_2;
            J(1, 5) = -fy*X/Z;

            H += J.transpose() * J;
            g += -J.transpose() * e;
        }

        //solve dx
        Vector6d dx;
        dx = H.ldlt().solve(g);
        T_esti = Sophus::SE3::exp(dx) * T_esti;

        if(isnan(dx[0])){
            cout << "result is nan!" << endl;
            break;
        }

        if(iter > 0 && cost >= lastCost){
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }


        lastCost = cost;
        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}