#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <sophus/se3.h>

using namespace std;
using namespace cv;

#include <pangolin/pangolin.h>

const string compare_file = "../compare.txt";
const string title_Before_registration = " Before registration ";
const string title_After_registration = " After registration ";

typedef Eigen::Matrix<double, 6, 1> Vector6d;

void read_data(const string& file_path, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
               vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te);
void DrawTrajectory(const string& title, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
                    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te);

void DrawTrajectory_1(const string& title, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
                     const vector<Point3d>& Pe);
void save_displacement_data(const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& T,
                            vector<Point3d>& P);

void pose_estimate_3d3d(const vector<Point3d>& Pg, const vector<Point3d>& Pe, Mat& R, Mat& t);

void translation_Tge_Pe(vector<Point3d>& Pg, Mat& R, Mat& t);

void check_read_data(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
                     vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te);

void check_displacement_data(vector<Point3d>& P);

int main(int argc, char** argv) {
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>  Tg;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>  Te;

    //read file, save data
    read_data(compare_file, Te, Tg);

    /*
     * check_read_data(Tg, Te);
     */

    //draw init picture
/*
 *   DrawTrajectory(title_Before_registration, Tg , Te);
 */

    vector<Point3d> Pg;
    vector<Point3d> Pe;

    //Recorver of displacement data information
    save_displacement_data(Tg, Pg);
    save_displacement_data(Te, Pe);
/*
 *   check_displacement_data(Pg);
 *   check_displacement_data(Pe);
 *
 */

    Mat R, t;
    pose_estimate_3d3d(Pg, Pe, R, t);

/*
    cout << "ICP via SVD results:" << endl;
    cout << "R_ = " << R << endl;
    cout << "t_ = " << t << endl;
    cout << "R_inv = " << R.t() << endl;        // R transpose()
    cout << "t_inv = " << t.t() << endl;        // t transpose()
*/
    //init value of pose estimation
    translation_Tge_Pe(Pe, R, t);
    //DrawTrajectory(title_After_registration, Tg , Te);
    DrawTrajectory_1(title_After_registration, Tg, Pe);
    return 0;
}

void pose_estimate_3d3d(const vector<Point3d>& Pg, const vector<Point3d>& Pe, Mat& R, Mat& t){
    assert(Pg.size() == Pe.size());
    Point3d p_g, p_e;
    int N = int(Pg.size());

    for(int i = 0; i < N; i++){
        p_g += Pg[i] / N;
        p_e += Pe[i] / N;
    }

    vector<Point3d> Qg, Qe;
    for(int i = 0; i < N; i++){
        Qg.push_back(Pg[i] - p_g);
        Qe.push_back(Qg[i] - p_e);
    }

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int j = 0; j < N; j++){
        Eigen::Vector3d q_g(Qg[j].x, Qg[j].y, Qg[j].z);
        Eigen::Vector3d q_e(Qe[j].x, Qe[j].y, Qe[j].z);
        W += q_g * q_e.transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d R_ = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Vector3d t_ = Eigen::Vector3d(p_g.x, p_g.y, p_g.z) - R_ * Eigen::Vector3d(p_e.x, p_e.y, p_e.z);

    //convert to cv
    R = (Mat_<double>(3, 3) <<
            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
            );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(0, 1), t_(0, 2));
}

void read_data(const string& file_path, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
               vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te){
    ifstream fin(file_path);
    if(!fin){
        cerr << "can't find" << file_path << endl;
        return;
    }
    string line;
    while(getline(fin, line)){
        std::stringstream stringstream1(line);
        double data[16] = {0};
        for(auto& d : data){
            stringstream1 >> d;
        }

        Eigen::Vector3d t_e(data[1], data[2], data[3]);
        Eigen::Quaterniond q_e;
        q_e.x() = data[4]; q_e.y() = data[5]; q_e.z() = data[6]; q_e.w() = data[7];
        Sophus::SE3 qt_e(q_e, t_e);
        Te.push_back(qt_e);

        Eigen::Vector3d t_g(data[9], data[10], data[11]);
        Eigen::Quaterniond q_g;
        q_g.x() = data[12]; q_g.y() = data[13]; q_g.z() = data[14]; q_g.w() = data[15];
        Sophus::SE3 qt_g(q_g, t_g);
        Tg.push_back(qt_g);
    }
}

void check_read_data(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
                     vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te){
    if(Tg.empty()){
        cerr << "Tg is empty " << endl;
        return;
    }
    if(Te.empty()){
        cerr << "Te is empty " << endl;
        return;
    }

    cout << "check tg " << endl;
    for(auto& tg : Tg){
        cout << tg.log().transpose() << endl;
    }

    cout << "check te " << endl;
    for(auto& te : Te){
        cout << te.log().transpose() << endl;
    }

}

void DrawTrajectory(const string& title, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
                    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te) {
    if (Tg.empty()) {
        cerr << "Tg is empty!" << endl;
        return;
    }
    if (Te.empty()){
        cerr << "Te is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < Tg.size() - 1; i++) {
            glColor3f(1 - (float) i / Tg.size(), 0.0f, (float) i / Tg.size());
            glBegin(GL_LINES);
            auto p1 = Tg[i], p2 = Tg[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        for (size_t i = 0; i < Te.size() - 1; i++) {
            glColor3f(1 - (float) i / Te.size(), 0.0f, (float) i / Te.size());
            glBegin(GL_LINES);
            auto p1 = Te[i], p2 = Te[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}

void DrawTrajectory_1(const string& title, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg,
                     const vector<Point3d>& Pe) {
    if (Tg.empty()) {
        cerr << "Tg is empty!" << endl;
        return;
    }
    if (Pe.empty()){
        cerr << "Pe is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < Tg.size() - 1; i++) {
            glColor3f(1 - (float) i / Tg.size(), 0.0f, (float) i / Tg.size());
            glBegin(GL_LINES);
            auto p1 = Tg[i], p2 = Tg[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        for (size_t i = 0; i < Pe.size() - 1; i++) {
            glColor3f(1 - (float) i / Pe.size(), 0.0f, (float) i / Pe.size());
            glBegin(GL_LINES);
            //auto p1 = Pe[i], p2 = Pe[i + 1];
            //glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            //glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glVertex3d(Pe[i].x, Pe[i].y, Pe[i].z);
            glVertex3d(Pe[i + 1].x, Pe[i + 1].y, Pe[i + 1].z);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}

void save_displacement_data(const vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& T,
                            vector<Point3d>& P){
    for(const auto t : T){
        Vector6d v = t.log();
        Point3d point3d;
        point3d.x = v[0]; point3d.y = v[1]; point3d.z = v[2];
        P.push_back(point3d);
    }
}

void check_displacement_data(vector<Point3d>& P){
    for(const auto& p : P){
        cout << p << endl;
    }
}

void translation_Tge_Pe(vector<Point3d>& Pe, Mat& R, Mat& t){
    assert(Pe.size() == Te_.size());
    int N = int(Pe.size());

    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    Eigen::Vector3d t_mat;
    t_mat << t.at<double>(0, 0), t.at<double>(0, 1), t.at<double>(0, 2);

    for(int i = 0; i < N; i++){
        Eigen::Vector3d Pe_prime = R_mat * Eigen::Vector3d(Pe[i].x, Pe[i].y, Pe[i].z) + t_mat;
        Pe[i].x = Pe_prime[0]; Pe[i].y = Pe_prime[1]; Pe[i].z = Pe_prime[2];
    }
}