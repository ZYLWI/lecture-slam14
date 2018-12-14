#include <iostream>
#include <sophus/se3.h>
#include <fstream>
#include <string>

#include <pangolin/pangolin.h>

using namespace std;

const string estimated_file = "../estimated.txt";
const string groundtruth_file = "../groundtruth.txt";

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>,
                    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

typedef Eigen::Matrix<double, 6, 1> Vector6d;

int main(int argc, char** argv) {
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> Te;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> Tg;

    ifstream fin_1(estimated_file);
    if(!fin_1){
        cerr << "can't find estimated_file" << endl;
        return -1;
    }
    string line_1;
    while(getline(fin_1, line_1)){
        std::stringstream stringstream1(line_1);
        double data[8] = {0};
        for(auto& d : data){
            stringstream1 >> d;
        }
        Eigen::Vector3d t(data[1], data[2], data[3]);
        Eigen::Quaterniond q;
        q.x() = data[4]; q.y() = data[5]; q.z() = data[6], q.w() = data[7];
        Sophus::SE3 qt(q, t);
        Te.push_back(qt);
    }

    ifstream fin_2(groundtruth_file);
    if(!fin_2){
        cout << "can't find groundtruth_file" << endl;
        return -1;
    }
    string line_2;
    while(getline(fin_2, line_2)){
        std::stringstream stringstream2(line_2);
        double data[8] = {0};
        for(auto& d : data){
            stringstream2 >> d;
        }
        Eigen::Vector3d t(data[1], data[2], data[3]);
        Eigen::Quaterniond q;
        q.x() = data[4]; q.y() = data[5]; q.z() = data[6], q.w() = data[7];
        Sophus::SE3 qt(q, t);
        Tg.push_back(qt);
    }

    assert(Te.size() == Tg.size());

    double e = 0.0;
    for(auto tg = Tg.begin(), te = Te.begin(); tg != Tg.end() && te != Te.end(); tg++, te++){
        Vector6d cost = ((*tg).inverse() * (*te)).log();
        e += cost.transpose() * cost;            //ei^2
    }

    double RMSE = sqrt(e / Tg.size());
    cout << RMSE << endl;
    DrawTrajectory(Tg, Te);

    return 0;
}

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> Tg,
                    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> Te) {
    if (Tg.empty() || Te.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
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