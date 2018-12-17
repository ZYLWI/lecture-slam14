#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <sophus/se3.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace cv;

#include <pangolin/pangolin.h>

const string compare_file = "../compare.txt";
const string title_Before_registration = "Before registration";
const string title_After_BA_registration = "After BA registration";

typedef Eigen::Matrix<double, 6, 1> Vector6d;

void read_data(const string& file_path, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te,
               vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg);

void pose_estimation_3d3d(const vector<Point3d>& Pg, const vector<Point3d>& Pe, Eigen::Matrix3d& R, Eigen::Vector3d& t);

void bundleAdjustment(const vector<Point3d>& Pg, const vector<Point3d>& Pe, Eigen::Matrix3d& R, Eigen::Vector3d& t);

void DrawTrajectory(const string& titile, vector<Point3d>& Pg, vector<Point3d>& Pe);

void translation_Pe(vector<Point3d>& Pe, Eigen::Matrix3d& R, Eigen::Vector3d& t);

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;
};

int main(int argc, char** argv) {
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>  Tg;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>  Te;

    read_data(compare_file, Te, Tg);

    vector<Point3d> Pg, Pe;
    //assert(Tg.size() == Te.size());
    for(int i = 0; i < Tg.size(); i++){
        Point3d pg;
        Eigen::Vector3d tg = Tg[i].translation();
        pg.x = tg[0]; pg.y = tg[1]; pg.z = tg[2];
        Pg.push_back(pg);

        Point3d pe;
        Eigen::Vector3d te = Te[i].translation();
        pe.x = te[0]; pe.y = te[1]; pe.z = te[2];
        Pe.push_back(pe);
    }

    DrawTrajectory(title_Before_registration, Pg, Pe);
    //first use SVD
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    pose_estimation_3d3d(Pg, Pe, R, t);


    cout << "SVD init" << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t.transpose() << endl;

    //then use Bundle Adjustment
    bundleAdjustment(Pg, Pe, R, t);

    cout << "Bundle Adjustment" << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t.transpose() << endl;

    translation_Pe(Pe, R, t);
    DrawTrajectory(title_After_BA_registration, Pg, Pe);
    return 0;
}
void read_data(const string& file_path, vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Te,
               vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>& Tg){
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

        //Emitated
        Eigen::Vector3d t_e(data[1], data[2], data[3]);
        Eigen::Quaterniond q_e;
        q_e.x() = data[4]; q_e.y() = data[5]; q_e.z() = data[6]; q_e.w() = data[7];
        Sophus::SE3 te(q_e, t_e);
        Te.push_back(te);

        //groundtruth
        Eigen::Vector3d t_g(data[9], data[10], data[11]);
        Eigen::Quaterniond q_g;
        q_g.x() = data[12]; q_g.y() = data[13]; q_g.z() = data[14]; q_g.w() =data[15];
        Sophus::SE3 tq(q_g, t_g);
        Tg.push_back(tq);
    }
}

void pose_estimation_3d3d(const vector<Point3d>& Pg, const vector<Point3d>& Pe, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    Point3d center_pg, center_pe;
    int N = int(Pg.size());
    for(int i = 0; i < N; i++){
        center_pg += Pg[i];
        center_pe += Pe[i];
    }
    center_pg = center_pg / N;
    center_pe = center_pe / N;

    vector<Point3d> Qg, Qe;
    for(int i = 0; i < N; i++){
        Qg.push_back(Pg[i] - center_pg);
        Qe.push_back(Pe[i] - center_pe);
    }

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0; i < N; i++){
        W += Eigen::Vector3d(Qg[i].x, Qg[i].y, Qg[i].z) * Eigen::Vector3d(Qe[i].x, Qe[i].y, Qe[i].z).transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    R = U*V.transpose();
    t = Eigen::Vector3d(center_pg.x, center_pg.y, center_pg.z) - R*Eigen::Vector3d(center_pe.x, center_pe.y, center_pe.z);
}

void bundleAdjustment(const vector<Point3d>& Pg, const vector<Point3d>& Pe, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    // init g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // pose dimension is 6, landmark dimension is 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d(0, 0, 0)
            ));
    optimizer.addVertex(pose);

    //edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly* > edges;
    for(size_t i = 0; i < Pg.size(); i++){
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(Pe[i].x, Pe[i].y, Pe[i].z)
                );
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement(Eigen::Vector3d(
                Pg[i].x, Pg[i].y, Pg[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity()*1e4);
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

   // cout << "T = " << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}

void DrawTrajectory(const string& title, vector<Point3d>& Pg, vector<Point3d>& Pe){
    if(Pg.empty()){
        cerr << "Tg is empty!" << endl;
        return;
    }

    if(Pe.empty()){
        cerr << "Te is empty!" << endl;
        return;
    }

    //create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
            );
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while(pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);

        for(size_t i = 0; i < Pe.size() - 1; i++){
            glColor3f(1 - (float)i / Pe.size(), 0.0f, (float)i / Pe.size());
            glBegin(GL_LINES);
            auto p1 = Pe[i], p2 = Pe[i + 1];
            glVertex3d(p1.x, p1.y, p1.z);
            glVertex3d(p2.x, p2.y, p2.z);
            glEnd();
        }

        for(size_t i = 0; i < Pg.size() - 1; i++){
            glColor3f(1 - (float)i / Pg.size(), 0.0f, (float)i / Pg.size());
            glBegin(GL_LINES);
            auto p1 = Pg[i], p2 = Pg[i + 1];
            glVertex3d(p1.x, p1.y, p1.z);
            glVertex3d(p2.x, p2.y, p2.z);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000); //sleep 5ms;
    }
}

void translation_Pe(vector<Point3d>& Pe, Eigen::Matrix3d& R, Eigen::Vector3d& t){
    int N = Pe.size();
    for(int i = 0; i < N; i++){
        Eigen::Vector3d pe = R * Eigen::Vector3d(Pe[i].x, Pe[i].y, Pe[i].z) + t;
        Pe[i].x = pe[0]; Pe[i].y = pe[1]; Pe[i].z = pe[2];
    }
}