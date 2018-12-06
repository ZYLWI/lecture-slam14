#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;

    vector<double> x_data, y_data;
    for(int i = 0; i < N; i++){
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma));
    }

    int iterations = 100;
    double cost = 0, lastCost = 0;
    for(int iter = 0; iter < iterations; iter++){
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        cost = 0;
        for(int i = 0; i < N; i++){
            double xi = x_data[i], yi = y_data[i];
            double error;
            error = yi - exp(ae*xi*xi + be*xi + ce);
            Vector3d J;
            J[0] = -1 * xi*xi * exp(ae*xi*xi + be*xi + ce);
            J[1] = -1 * xi * exp(ae*xi*xi + be*xi + ce);
            J[2] = -1 * exp(ae*xi*xi + be*xi + ce);

            H += J * J.transpose();
            b += -error * J;

            cost += error*error;
        }
        Vector3d dx = H.ldlt().solve(b);
        if(isnan(dx[0])){
            cout << "result is nan!" << endl;
        }
        if(iter > 0 && cost > lastCost){
            cout << "cost:" << cost << ",last cost:" << lastCost << endl;
            break;
        }
        //update abc
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        cout << "total cost:" << cost << endl;
    }
    cout << "estimated abc =" << ae <<", " << be << ", " << ce << endl;
    return 0;
}