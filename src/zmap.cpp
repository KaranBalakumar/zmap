#include <librealsense2/rs.hpp>
#include <rerun.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

// #include <rerun/archetypes/transform3d.hpp>
// #include <rerun/components/translation3d.hpp>
// #include <rerun/components/rotation_quat.hpp>
// #include <rerun/datatypes/quaternion.hpp>

Eigen::Matrix4f load_extrinsics(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Couldn't open " + filepath);
    }

    Eigen::Matrix4f H = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ifs >> H(j, i);  // Read as column-major
        }
    }

    float det = H.block<3,3>(0,0).determinant();
    if (std::abs(det - 1.0f) > 1e-6) {
        throw std::runtime_error("Invalid transformation matrix: det != 1");
    }

    return H;
}

int main() try {
    const auto rec = rerun::RecordingStream("realsense_depth");
    rec.spawn().exit_on_failure();
    rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN);

    rs2::context ctx;
    rs2::pipeline pipe_d435(ctx), pipe_t265(ctx);

    rs2::config cfg_d435, cfg_t265;
    cfg_d435.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg_t265.enable_stream(RS2_STREAM_POSE);

    pipe_d435.start(cfg_d435);
    pipe_t265.start(cfg_t265);

    rs2::frameset frames = pipe_d435.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();

    rs2::video_stream_profile profile = depth.get_profile().as<rs2::video_stream_profile>();
    rs2_intrinsics intr = profile.get_intrinsics();

    rec.log(
        "world/camera",
        rerun::Pinhole::from_focal_length_and_resolution(
            {intr.fx, intr.fy},
            {static_cast<float>(intr.width), static_cast<float>(intr.height)}
        )
    );

    Eigen::Matrix4f H_t265_d400 = load_extrinsics("/home/karan/zmap/H_t265_d400.cfg");
    std::vector<Eigen::Vector3f> trajectory_points;

    std::cout << "Streaming with T265 pose + D435 depth... Press Ctrl+C to stop." << std::endl;

    while (true) {
        auto frames_d435 = pipe_d435.wait_for_frames();
        auto frames_t265 = pipe_t265.wait_for_frames();

        rs2::depth_frame depth = frames_d435.get_depth_frame();
        rs2::pose_frame pose_frame = frames_t265.get_pose_frame();

        if (!depth || !pose_frame) continue;

        auto pose_data = pose_frame.get_pose_data();

        // Construct 4x4 pose matrix from T265
        Eigen::Matrix4f T_t265 = Eigen::Matrix4f::Identity();
        T_t265.block<3,3>(0,0) = Eigen::Quaternionf(
            -pose_data.rotation.w,
            -pose_data.rotation.z,
            pose_data.rotation.y,
            pose_data.rotation.x
        ).toRotationMatrix();
        T_t265.block<3,1>(0,3) = Eigen::Vector3f(
            -pose_data.translation.x,
            -pose_data.translation.y,
            pose_data.translation.z
        );

        // Transform to D435 pose
        Eigen::Matrix4f T_d435 = T_t265 * H_t265_d400;

        Eigen::Vector3f position = T_d435.block<3,1>(0,3);
        Eigen::Quaternionf orientation(T_d435.block<3,3>(0,0));

        // Convert to rerun components
        rerun::components::Translation3D translation{position.x(), position.y(), position.z()};
        // rerun::components::RotationQuat rotation{orientation.w(), orientation.x(), orientation.y(), orientation.z()};

        rerun::datatypes::Quaternion quat{
            orientation.w(), orientation.x(), orientation.y(), orientation.z()
        };
        rerun::components::RotationQuat rotation{quat};

        // Log camera pose
        rec.log(
            "world/camera",
            rerun::Transform3D::from_translation_rotation(translation, rotation)
        );

        // Log depth image
        const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth.get_data());
        uint32_t width = static_cast<uint32_t>(depth.get_width());
        uint32_t height = static_cast<uint32_t>(depth.get_height());

        rec.log(
            "world/camera/depth",
            rerun::DepthImage(depth_data, {width, height})
                .with_meter(1000.0f)
                .with_colormap(rerun::components::Colormap::Viridis)
        );

        // log trajectory
        // trajectory_points.push_back(position);
        // // Convert trajectory to rerun-friendly format
        // std::vector<rerun::datatypes::Vec3D> rerun_trajectory;
        // rerun_trajectory.reserve(trajectory_points.size());
        // for (const auto& pt : trajectory_points) {
        //     rerun_trajectory.emplace_back(pt.x(), pt.y(), pt.z());
        //     // std::cout << "Trajectory point: " << pt.x() << ", " << pt.y() << ", " << pt.z() << std::endl;
        // }

        // // Log the trajectory as a line strip
        // rec.log("world/camera/trajectory", rerun::archetypes::LineStrips3D(rerun_trajectory));
    }

    return EXIT_SUCCESS;

} catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.what() << std::endl;
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
}