#include <librealsense2/rs.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>
#include <rerun.hpp>
#include <rerun/time.hpp>
// #include <rerun/archetypes/line_strip3d.hpp>

float detR(float H[16]) {
    return H[0]*(H[5]*H[10]-H[9]*H[6]) - H[4]*(H[1]*H[10]-H[2]*H[9]) + H[8]*(H[1]*H[6]-H[5]*H[2]);
}

Eigen::Matrix3d reorthogonalize(const Eigen::Matrix3d &R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

// void draw_pointcloud_rerun(rs2::points& points, const rs2_pose& pose, float H_t265_d400[16],
//                            std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pointcloud,
//                            const rerun::RecordingStream& rec) {
//     if (!points)
//         return;

//     Eigen::Quaterniond q(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z);
//     Eigen::Vector3d t(pose.translation.x, pose.translation.y, pose.translation.z);
//     Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
//     T.rotate(q);
//     T.pretranslate(t);

//     Eigen::Matrix3d R_;
//     R_ << H_t265_d400[0], H_t265_d400[1], H_t265_d400[2],
//           H_t265_d400[4], H_t265_d400[5], H_t265_d400[6],
//           H_t265_d400[8], H_t265_d400[9], H_t265_d400[10];
//     R_ = reorthogonalize(R_);
//     Eigen::Vector3d t_(H_t265_d400[3], H_t265_d400[7], H_t265_d400[11]);
//     Eigen::Isometry3d T_ = Eigen::Isometry3d::Identity();
//     T_.rotate(R_);
//     T_.pretranslate(t_);

//     pointcloud.clear();
//     const auto vertices = points.get_vertices();
//     size_t count = points.size();

//     std::vector<rerun::Position3D> rerun_points;
//     rerun_points.reserve(count);

//     for (size_t i = 0; i < count; i++) {
//         const rs2::vertex& vertex = vertices[i];
//         Eigen::Vector3d p(vertex.x, vertex.y, vertex.z);
//         Eigen::Vector3d pw = T * T_ * p;
//         rerun_points.emplace_back(pw[0], pw[1], pw[2]);
//         pointcloud.push_back(pw);
//     }

//     rec.log("pointcloud", rerun::Points3D(rerun_points));
// }

void draw_pointcloud_rerun(rs2::points& points, const rs2_pose& pose, float H_t265_d400[16],
                           std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pointcloud,
                           rerun::RecordingStream& rec) {
    if (!points)
        return;

    Eigen::Quaterniond q(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z);
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

    std::vector<rerun::Position3D> rerun_points;
    rerun_points.reserve(count);

    for (size_t i = 0; i < count; i++) {
        const rs2::vertex& vertex = vertices[i];
        Eigen::Vector3d p(vertex.x, vertex.y, vertex.z);
        Eigen::Vector3d pw = T * T_ * p;
        rerun_points.emplace_back(pw[0], pw[1], pw[2]);
        pointcloud.push_back(pw);
    }

    // This line now avoids memory growth
    // rec.log_timeless("pointcloud", rerun::Points3D(rerun_points));
    rec.log("pointcloud", rerun::Points3D(rerun_points), rerun::TimePoint::sequence(0));
}

int main(int argc, char * argv[]) try {
    const auto rec = rerun::RecordingStream("realsense_pointcloud");
    rec.spawn().exit_on_failure();

    rs2::pointcloud pc;
    rs2::points points;
    rs2::pose_frame pose_frame(nullptr);
    std::vector<rs2_vector> trajectory;

    rs2::context ctx;
    std::map<std::string, rs2::colorizer> colorizers;
    std::vector<rs2::pipeline> pipelines;

    std::vector<std::string> serials;
    for (auto&& dev : ctx.query_devices())
        serials.push_back(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

    for (auto&& serial : serials) {
        rs2::pipeline pipe(ctx);
        rs2::config cfg;
        cfg.enable_device(serial);
        pipe.start(cfg);
        pipelines.emplace_back(pipe);
        colorizers[serial] = rs2::colorizer();
    }

    float H_t265_d400[16] =  {1, 0, 0, 0,
                               0,-1, 0, 0,
                               0, 0,-1, 0,
                               0, 0, 0, 1};
    std::ifstream ifs("./H_t265_d400.cfg");
    if (!ifs.is_open()) {
        std::cerr << "Couldn't open H_t265_d400.cfg" << std::endl;
        return -1;
    } else {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                ifs >> H_t265_d400[i+4*j];
            }
        }
    }
    if (fabs(1 - detR(H_t265_d400)) > 1e-6) {
        std::cerr << "Invalid transformation matrix (det != 1)" << std::endl;
        return -1;
    }

    Eigen::Matrix<double, 3, 1> p;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointcloud;
    pointcloud.reserve(1000000);

    while (true) {
        for (auto &&pipe : pipelines) {
            auto frames = pipe.wait_for_frames();
            auto depth = frames.get_depth_frame();
            if (depth)
                points = pc.calculate(depth);

            auto pose = frames.get_pose_frame();
            if (pose) {
                pose_frame = pose;
                auto pose_data = pose.get_pose_data();
                std::cout << "\rDevice Position: " << std::fixed << std::setprecision(3)
                          << pose_data.translation.x << " "
                          << pose_data.translation.y << " "
                          << pose_data.translation.z << " (m)";

                if (trajectory.empty() ||
                    sqrt(pow(pose_data.translation.x - trajectory.back().x, 2) +
                         pow(pose_data.translation.y - trajectory.back().y, 2) +
                         pow(pose_data.translation.z - trajectory.back().z, 2)) > 0.002) {
                    trajectory.push_back(pose_data.translation);
                }
            }
        }

        if (points && pose_frame) {
            rs2_pose pose = pose_frame.get_pose_data();
            draw_pointcloud_rerun(points, pose, H_t265_d400, pointcloud, rec);
            // draw_pointcloud_rerun(points, pose_frame.get_pose_data(), H_t265_d400, pointcloud, rec);

            // Optional: Log trajectory
            std::vector<rerun::Position3D> traj;
            for (const auto& v : trajectory)
                traj.emplace_back(v.x, v.y, v.z);
            // rec.log("trajectory", rerun::LineStrip3D(traj));
            rec.log("trajectory", rerun::archetypes::LineStrips3D(traj));
        }
    }

    return EXIT_SUCCESS;
} catch (const rs2::error & e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
