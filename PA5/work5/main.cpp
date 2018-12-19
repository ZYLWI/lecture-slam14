//
// Created by yu on 18-12-18.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <chrono>

using namespace std;
using namespace cv;

const string img1_path = "../1.png";
const string img2_path = "../2.png";
const Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
const double cx = 325.1;
const double cy = 249.7;
const double fx = 520.9;
const double fy = 521.0;

void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches);

void bundleAdjustment(const vector<Point2f>& Pc1, const vector<Point2f>& Pc2);

class DrawKeyPoints : public G2O_TYPES_SBA_API g2o::EdgeProjectXYZ2UV{
        void print(){
            cout << _error << endl;
        }
};

int main(int argc, char** argv){
    // read picture
    Mat img1 = imread(img1_path, CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(img2_path, CV_LOAD_IMAGE_COLOR);

    if(img1.empty() || img2.empty()){
        cerr << "can't find img" << endl;
        return 1;
    }

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints_1, keypoints_2, matches);

    vector<Point2f> Pc1, Pc2;
    for(int i = 0; i < matches.size(); i++){
        Pc1.push_back(keypoints_1[matches[i].queryIdx].pt);
        Pc2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    bundleAdjustment(Pc1, Pc2);
    return 0;
}

void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,std::vector<DMatch>& matches){
    //init
    Mat descriptors_1, descriptors_2;
    //used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> mathcer = DescriptorMatcher::create("BruteForce-Hamming");

    //fisrt, Detect Oriented FAST Corner Position
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //second, Calculate the brief descriptor from the corner position
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //third, The BRIEF descriptors in the two images are matched using hamming distance
    vector<DMatch> match;
    mathcer->match(descriptors_1, descriptors_2, match);

    //fourth, match point pair filtering
    double min_dist = 10000, max_dist = 0;

    //Find out the minimum distance and maximum distance between all matches, that is,
    // the distance between the most similar and the least similar two groups of points

    for(int i = 0; i < descriptors_1.rows; i++){
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    //When the distance between descriptors is greater than twice the minimum distance,
    // the match is considered incorrect. However, sometimes the minimum distance will
    // be very small and an empirical value of 30 will be set as the lower limit.

    for(int i = 0; i < descriptors_1.rows; i++){
        if(match[i].distance <= max(2 * min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
}

void bundleAdjustment(const vector<Point2f>& Pc1, const vector<Point2f>& Pc2){
    // init g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // pose dimension is 6, landmark dimension is 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    //add vertex
    for(int i = 0; i < 2; i++){
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if(i == 0)
            v->setFixed(true);
        v->setEstimate(g2o::SE3Quat());
        optimizer.addVertex(v);
    }

    for(size_t i = 0; i < Pc1.size(); i++){
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId(2 + i);
        double z = 1;
        double x = (Pc1[i].x - cx) * z / fx;
        double y = (Pc1[i].y - cy) * z / fy;
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y , z));
        optimizer.addVertex(v);
    }

    // camera
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    //img1
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for(int i = 0; i < Pc1.size(); i++){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(0)));
        edge->setMeasurement(Eigen::Vector2d(Pc1[i].x, Pc1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);

        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    //img2
    for(int i = 0; i < Pc2.size(); i++){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(1)));
        edge->setMeasurement(Eigen::Vector2d(Pc2[i].x, Pc2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    cout << " start optimizer" << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    cout << "end optimizer" << endl;

    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
    Eigen::Isometry3d pose = v->estimate();
    cout << "Pose = " << endl << pose.matrix() << endl;
}
