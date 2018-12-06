//
// Created by 高翔 on 2017/12/15.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

const string image_file = "../test.png";

int main(int argc, char **argv) {

    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;

    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0);
    if(image.empty())   cout << "can't find image" << endl;
    int rows = image.rows, cols = image.cols;

    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);

    for(int v = 0; v < rows; v++){
        for(int u = 0; u < cols; u++){
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;

            double u_distorted = 0, v_distorted = 0;
            double r2 = pow(x, 2) + pow(y, 2);
            double r4 = pow(r2, 2);

            double x_distorted = x*(1 + k1*r2 + k2*r4) + 2*p1*x*y + p2*(r2 + 2*x*x);
            double y_distorted = y*(1 + k1*r2 + k2*r4) + 2*p2*x*y + p1*(r2 + 2*y*y);

            u_distorted = x_distorted * fx + cx;
            v_distorted = y_distorted * fy + cy;

            if(u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows){
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int)v_distorted, (int)u_distorted);
            }else{
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }
    cv::imshow("image undistorted", image_undistort);
    cv::waitKey();
    return 0;
}