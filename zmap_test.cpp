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
    Sophus::Vector3d t(pose.translation.x, pose.translation.y, -1.0 * pose.translation.z);
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

    rs2::context                          ctx;        // Create librealsense context for managing devices
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
        // Map from each device's serial number to a different colorizer
        // colorizers[serial] = rs2::colorizer();
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

    // Eigen::Matrix3d R1;
    // R1(0, 0) = H_t265_d400[0]; R1(0, 1) = H_t265_d400[1]; R1(0, 2) = H_t265_d400[2];
    // R1(1, 0) = H_t265_d400[4]; R1(1, 1) = H_t265_d400[5]; R1(1, 2) = H_t265_d400[6];
    // R1(2, 0) = H_t265_d400[8]; R1(2, 1) = H_t265_d400[9]; R1(1, 2) = H_t265_d400[10];

    // Eigen::Vector3d t__(H_t265_d400[3], H_t265_d400[7], H_t265_d400[11]);

    // Sophus::SE3d T_t265_d400(R1, t__);

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
                    glBegin(GL_LINES);
                    glColor3f(255.0, 0.0, 0.0);
                    glVertex3d(prev.x, prev.y, prev.z);
                    glVertex3d(curr.x, curr.y, curr.z);
                    // for(size_t i=0; i < trajectory.size(); i++){
                    //     auto p1 = trajectory[i], p2 = trajectory[i+1];
                    //     glVertex3d(p1.x, p1.y, p1.z);
                    //     glVertex3d(p2.x, p2.y, p2.z);
                    // }
                    glColor3f(255.0, 255.0, 255.0);
                    glEnd();   
                }
                // std::cout << T.matrix3x4() << std::endl;
            }
        }

        // Clear the previous point cloud before adding new points
        if (points && pose_frame) {
            pointcloud.clear(); // Clear previous points

            // Get the number of points
            const auto vertices = points.get_vertices();
            size_t count = points.size();

            for (size_t i = 0; i < count; i++) {
                const rs2::vertex& vertex = vertices[i]; // Corrected type
                p[0] = vertex.x;
                p[1] = vertex.y;
                p[2] = vertex.z;
                // Eigen::Vector3d p1;
                // p1[0] = p[0];
                // p1[1] = p[1];
                // p1[2] = p[2];
                Eigen::Vector3d pointWorld = T * p;
                pointcloud.push_back(pointWorld); // Store transformed point
            }

            // Activate the camera and render the new point cloud
            glPointSize(0.025);
            // s_cam.Follow(m, true);
            glBegin(GL_POINTS);
            for (const auto &point : pointcloud) {
                glVertex3d(point[0], point[1], point[2]);
            }
            glEnd();
            pangolin::FinishFrame();
        }

        // Optional: Add a sleep to control the output rate
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return EXIT_SUCCESS;
}