#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "include/render.hpp"          // Include short list of convenience functions for rendering
#include <algorithm>            // std::min, std::max
#include <fstream>              // std::ifstream
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

void draw_pointcloud_in_world(float width, float height, glfw_state& app_state, rs2::points& points, rs2_pose& pose, float H_t265_d400[16], std::vector<rs2_vector>& trajectory,
                                Eigen::Matrix<double, 3, 1> &p, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pointcloud);

float detR(float H[16]) {
    return H[0]*(H[5]*H[10]-H[9]*H[6]) - H[4]*(H[1]*H[10]-H[2]*H[9]) + H[8]*(H[1]*H[6]-H[5]*H[2]);
}

Eigen::Matrix3d reorthogonalize(const Eigen::Matrix3d &R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

int d = 0.05;
int rows=100;
int cols=100;
std::vector<std::vector<double>> grid(rows, std::vector<double>(cols, 0));
std::vector<std::vector<int>> gridc(rows, std::vector<int>(cols, 0));
std::vector<std::vector<double>> GRID(rows, std::vector<double>(cols, 0));

int main(int argc, char * argv[]) try
{
    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "zmap demo");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;
    // store pose and timestamp
    rs2::pose_frame pose_frame(nullptr);
    std::vector<rs2_vector> trajectory;

    rs2::context                          ctx;        // Create librealsense context for managing devices
    std::map<std::string, rs2::colorizer> colorizers; // Declare map from device serial number to colorizer (utility class to convert depth data RGB colorspace)
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
        colorizers[serial] = rs2::colorizer();
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
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointcloud;
    pointcloud.reserve(1000000);

    while (app) // Application still alive?
    {
        for (auto &&pipe : pipelines) // loop over pipelines
        {
            // Wait for the next set of frames from the camera
            auto frames = pipe.wait_for_frames();
            auto depth = frames.get_depth_frame();

            // Generate the pointcloud and texture mappings
            if (depth)
                points = pc.calculate(depth);

            // pose
            auto pose = frames.get_pose_frame();
            if (pose) {
                pose_frame = pose;

                // Print the x, y, z values of the translation, relative to initial position
                auto pose_data = pose.get_pose_data();
                std::cout << "\r" << "Device Position: " << std::setprecision(3) << std::fixed << pose_data.translation.x << " " << pose_data.translation.y << " " << pose_data.translation.z << " (meters)";

                // add new point in the trajectory (if motion large enough to reduce size of traj. vector)
                if (trajectory.size() == 0)
                    trajectory.push_back(pose_data.translation);
                else {
                    rs2_vector prev = trajectory.back();
                    rs2_vector curr = pose_data.translation;
                    if (sqrt(pow((curr.x - prev.x), 2) + pow((curr.y - prev.y), 2) + pow((curr.z - prev.z), 2)) > 0.002)
                        trajectory.push_back(pose_data.translation);
                }
             }
        }

        // Draw the pointcloud
        if (points && pose_frame) {
            rs2_pose pose =  pose_frame.get_pose_data();
            draw_pointcloud_in_world(app.width(), app.height(), app_state, points, pose, H_t265_d400, trajectory, p, pointcloud);
        }
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

void draw_pointcloud_in_world(float width, float height, glfw_state& app_state, rs2::points& points, rs2_pose& pose, float H_t265_d400[16], std::vector<rs2_vector>& trajectory,
                                Eigen::Matrix<double, 3, 1> &p, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &pointcloud){
    if (!points)
        return;

    // OpenGL commands that prep screen for the pointcloud
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);


    // viewing matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // rotated from depth to world frame: z => -z, y => -y
    glTranslatef(0, 0, -0.75f - app_state.offset_y * 0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, -1, 0);
    glTranslatef(0, 0, 0.5f);

    // draw trajectory
    glEnable(GL_DEPTH_TEST);
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (auto&& v : trajectory)
    {
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(v.x, v.y, v.z);
    }
    glEnd();
    glLineWidth(0.5f);
    glColor3f(1.0f, 1.0f, 1.0f);

    Eigen::Quaterniond q;
    q.w() = pose.rotation.w;
    q.x() = pose.rotation.x;
    q.y() = pose.rotation.y;
    q.z() = pose.rotation.z;
    Eigen::Vector3d t(pose.translation.x, pose.translation.y, pose.translation.z);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(q);
    T.pretranslate(t);


    Eigen::Matrix3d R_;
    R_ << H_t265_d400[0], H_t265_d400[1], H_t265_d400[2],
          H_t265_d400[4], H_t265_d400[5], H_t265_d400[6],
          H_t265_d400[8], H_t265_d400[9], H_t265_d400[10];
    R_ = reorthogonalize(R_);
    Eigen::Vector3d t_(H_t265_d400[3], H_t265_d400[7], H_t265_d400[11]);
    Eigen::Isometry3d T_ = Eigen::Isometry3d::Identity();
    T_.rotate(R_);
    T_.pretranslate(t_);

    pointcloud.clear();
    const auto vertices = points.get_vertices();
    size_t count = points.size();

    for (size_t i = 0; i < count; i++) {
        const rs2::vertex& vertex = vertices[i]; // Corrected type
        p[0] = vertex.x;
        p[1] = vertex.y;
        p[2] = vertex.z;

        Eigen::Vector3d pointWorld = T_ * p;
        pointWorld = T * pointWorld;
        grid[pointWorld[0]/d][pointWorld[1]/d] += pointWorld[2];
        gridc[pointWorld[0]/d][pointWorld[1]/d] += 1;
        pointcloud.push_back(pointWorld); // Store transformed point
    }

    // glPointSize(width / 640);
    glPointSize(0.005);
    glColor3f(0.0, 0.0, 0.0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_POINTS);
    double sum=0;

    for (const auto &point : pointcloud) {
        glVertex3d(point[0], point[1], point[2]);
    }

    // OpenGL cleanup
    glEnd();

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            GRID[i][j] = grid[i][j] / gridc[i][j];
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
}

