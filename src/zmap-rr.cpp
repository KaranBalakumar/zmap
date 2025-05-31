#include <librealsense2/rs.hpp>
#include <rerun.hpp>
#include <iostream>
#include <vector>
#include <cstring>

int main() try {
    // Start Rerun logging stream
    const auto rec = rerun::RecordingStream("realsense_depth");
    rec.spawn().exit_on_failure();

    // Set up RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    // Wait for first frame to extract intrinsics
    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();

    rs2::video_stream_profile profile = depth.get_profile().as<rs2::video_stream_profile>();
    rs2_intrinsics intr = profile.get_intrinsics();

    // Log pinhole camera model (only once)
    rec.log(
        "camera",
        rerun::Pinhole::from_focal_length_and_resolution(
            {intr.fx, intr.fy},
            {static_cast<float>(intr.width), static_cast<float>(intr.height)}
        )
    );

    std::cout << "Streaming depth... Press Ctrl+C to stop." << std::endl;

    while (true) {
        frames = pipe.wait_for_frames();
        depth = frames.get_depth_frame();

        int width = depth.get_width();
        int height = depth.get_height();
        const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth.get_data());

        // Log depth image normally (each frame is a new frame in Rerun)
        rec.log(
            "camera/depth",
            rerun::DepthImage(depth_data, {width, height})
                .with_meter(1000.0f)  // Convert mm to meters
                .with_colormap(rerun::components::Colormap::Viridis)
        );
    }

    return EXIT_SUCCESS;

} catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.what() << std::endl;
    return EXIT_FAILURE;

} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
