#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

Eigen::Matrix3d reorthogonalize(const Eigen::Matrix3d &R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

Sophus::SE3d getT(const auto &pose, const auto &T_t265_d400){
    Eigen::Quaterniond q;
    Sophus::Vector3d t(pose.translation.x, pose.translation.y, pose.translation.z);
    q.w() = pose.rotation.w;
    q.x() = pose.rotation.x;
    q.y() = pose.rotation.y;
    q.z() = pose.rotation.z;
    Sophus::SE3d T(q,t);
    // Eigen::Matrix3d R_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0,1,0)).toRotationMatrix();
    Eigen::Matrix3d R_;
    R_ << T_t265_d400[0], T_t265_d400[1], T_t265_d400[2]
        , T_t265_d400[4], T_t265_d400[5], T_t265_d400[6]
        , T_t265_d400[8], T_t265_d400[9], T_t265_d400[10];
    R_ = reorthogonalize(R_);
    Eigen::Vector3d t_(T_t265_d400[3], T_t265_d400[7], T_t265_d400[11]);
    Sophus::SE3d Tconv(R_, t_);
    return  T*Tconv;

}

float detR(float H[16]) {
    return H[0]*(H[5]*H[10]-H[9]*H[6]) - H[4]*(H[1]*H[10]-H[2]*H[9]) + H[8]*(H[1]*H[6]-H[5]*H[2]);
}

double d = 0.05;
int rows=100;
int cols=100;
std::vector<std::vector<double>> grid(rows, std::vector<double>(cols, 0));
std::vector<std::vector<int>> gridc(rows, std::vector<int>(cols, 0));
std::vector<std::vector<double>> GRID(rows, std::vector<double>(cols, 0));

int main(int argc, char * argv[])   
{
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;
    // store pose and timestamp
    rs2::pose_frame pose_frame(nullptr);

    std::vector<rs2_vector> trajectory;

    rs2::context ctx;        // Create librealsense context for managing devices
    // std::map<std::string, rs2::colorizer> colorizers; // Declare map from device serial number to colorizer (utility class to convert depth data RGB colorspace)
    std::vector<rs2::pipeline>            pipelines;

    // Capture serial numbers before opening streaming
    std::vector<std::string>              serials;
    for (auto&& dev : ctx.query_devices())
        serials.push_back(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));


    // Start a streaming pipe per each connected device
    for (auto&& serial : serials)
    {
        rs2::pipeline pipe(ctx);
        rs2::config cfg;
        cfg.enable_device(serial);
        pipe.start(cfg);
        pipelines.emplace_back(pipe);
    }

    // extrinsics
    // depth w.r.t. tracking (column-major)
    float H_t265_d400[16] =  {1, 0, 0, 0,
                              0,-1, 0, 0,
                              0, 0,-1, 0,
                              0, 0, 0, 1};
    std::string fn = "./H_t265_d400.cfg";
    std::ifstream ifs(fn);
    if (!ifs.is_open()) {
        std::cerr << "Couldn't open " << fn << std::endl;
        return -1;
    }
    else {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                ifs >> H_t265_d400[i+4*j];  // row-major to column-major
            }
        }
    }
    float det = detR(H_t265_d400);
    if (fabs(1-det) > 1e-6) {
        std::cerr << "Invalid homogeneous transformation matrix input (det != 1)" << std::endl;
        return -1;
    }

    Eigen::Matrix<double, 3, 1> p;
    std::vector<Sophus::Vector3d, Eigen::aligned_allocator<Sophus::Vector3d>> pointcloud;
    pointcloud.reserve(1000000);
    Sophus::SE3d T;

    while (pangolin::ShouldQuit() == false) // Application still alive?
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        
        for (auto &&pipe : pipelines) // loop over pipelines
        {
            // Wait for the next set of frames from the camera
            auto frames = pipe.wait_for_frames();

            auto depth = frames.get_depth_frame();

            // Generate the pointcloud and texture mappings
            if (depth)
                points = pc.calculate(depth);

            // std::cout << points.size() << std::endl;

            auto pose = frames.get_pose_frame();
            if(pose){
                pose_frame = pose;
                auto pose_data = pose.get_pose_data();
                T = getT(pose_data, H_t265_d400);
                if(trajectory.size() == 0){
                    trajectory.push_back(pose_data.translation);
                }else{
                    rs2_vector prev = trajectory.back();
                    rs2_vector curr = pose_data.translation;
                    trajectory.push_back(curr);
                    glEnable(GL_DEPTH_TEST);
                    glLineWidth(2.0f);
                    glBegin(GL_LINE_STRIP);
                    for (auto&& v : trajectory)
                    {
                        glColor3f(0.0f, 1.0f, 0.0f);
                        glVertex3f(v.x, v.y, v.z);
                    }
                    glEnd();
                    glColor3f(1.0f, 1.0f, 1.0f);  
                }
            }
        }

        // Clear the previous point cloud before adding new points
        if (points && pose_frame) {
            pointcloud.clear(); // Clear previous points

            // Get the number of points
            const auto vertices = points.get_vertices();
            size_t count = points.size();
            int x_index, y_index;

            for (size_t i = 0; i < count; i++) {
                const rs2::vertex& vertex = vertices[i]; // Corrected type
                p[2] = -1.0 * vertex.x;
                p[1] = vertex.y;
                p[0] = vertex.z;

                Eigen::Vector3d pointWorld = T * p;
                
                // std::cout << pointWorld[0] / d << " " << pointWorld[1] / d  << std::endl;
                x_index = int(double(pointWorld[0])/d) + int(rows/2);
                y_index = int(pointWorld[1]/d) + int(cols/2);
                // std::cout << x_index << " " << y_index << std::endl;
                // std::cout << "X - " << pointWorld[0]/d << " - " << int(pointWorld[0]/d) << std::endl;
                if (x_index >= 0 && x_index < rows && y_index >= 0 && y_index < cols) {
                    grid[x_index][y_index] += pointWorld[2];
                    gridc[x_index][y_index]++;
                }
                pointcloud.push_back(pointWorld); // Store transformed point
            }
            Eigen::Vector3d pointWorld = T * p;
            std::ofstream file("/home/karan/zmap/data.txt");
            
            
            // Activate the camera and render the new point cloud
            glPointSize(0.025);
            glBegin(GL_POINTS);
            for (const auto &point : pointcloud) {
                glVertex3d(point[0], point[1], point[2]);
                file << point[0] << " " << point[1] << " " << point[2] << "\n";
            }
            file.close();
            
            glEnd();
            pangolin::FinishFrame();

            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    if(gridc[i][j] > 0)
                        GRID[i][j] = double(grid[i][j] / double(gridc[i][j]));
                    // std::cout << i << " " << j << " " << grid[i][j]<< " / " << gridc[i][j] << " = " << GRID[i][j] << std::endl;
                }
            }

            // std::ofstream file("/home/karan/zmap/data.txt");
            // for (int i = 0; i < rows; i++) {
            //     for (int j = 0; j < cols; j++) {
            //         file << i << " " << j << " " << GRID[i][j] << "\n";
            //     }
            // }
            // file.close();
        }
    }

    return EXIT_SUCCESS;
}