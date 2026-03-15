#include <algorithm>
#include <cctype>
#include <dirent.h>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace {

std::string basenameNoExt(const std::string &path) {
    const size_t slash = path.find_last_of("/\\");
    const std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

bool isDigits(const std::string &s) {
    return !s.empty() &&
           std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

int numericStem(const std::string &path) {
    const std::string stem = basenameNoExt(path);
    if (!isDigits(stem)) return -1;
    return std::stoi(stem);
}

std::vector<std::string> collectPcdFiles(const std::string &dir_path) {
    std::vector<std::string> files;
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) return files;

    dirent *ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name(ent->d_name);
        if (name == "." || name == "..") continue;
        if (name.size() < 4) continue;

        std::string ext = name.substr(name.size() - 4);
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext != ".pcd") continue;

        std::string full_path = dir_path + "/" + name;
        struct stat st {};
        if (stat(full_path.c_str(), &st) != 0) continue;
        if (!S_ISREG(st.st_mode)) continue;
        files.push_back(full_path);
    }
    closedir(dir);

    std::sort(files.begin(), files.end(), [](const std::string &a, const std::string &b) {
        const int na = numericStem(a);
        const int nb = numericStem(b);
        if (na >= 0 && nb >= 0 && na != nb) return na < nb;
        if (na >= 0 && nb < 0) return true;
        if (na < 0 && nb >= 0) return false;
        return a < b;
    });

    return files;
}

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "patchworkpp_pcd_player");
    ros::NodeHandle pnh("~");

    std::string pcd_dir;
    std::string topic;
    std::string frame_id;
    int map_test_hz = 10;
    int start_idx = 0;
    bool step_mode = false;
    bool latch = true;

    pnh.param<std::string>("pcd_dir", pcd_dir, "/home/ros/gTRAVEL_ws/src/map_test/pcd");
    pnh.param<std::string>("topic", topic, "/patchworkpp/input_cloud");
    pnh.param<std::string>("frame_id", frame_id, "map");
    pnh.param<int>("map_test_hz", map_test_hz, 10);
    pnh.param<int>("start_idx", start_idx, 0);
    pnh.param<bool>("stop", step_mode, false);
    pnh.param<bool>("latch", latch, true);

    std::vector<std::string> pcd_files = collectPcdFiles(pcd_dir);
    if (pcd_files.empty()) {
        ROS_ERROR_STREAM("No .pcd files found in: " << pcd_dir);
        return 1;
    }
    if (start_idx < 0 || start_idx >= static_cast<int>(pcd_files.size())) {
        ROS_ERROR_STREAM("start_idx out of range: " << start_idx << ", total files: " << pcd_files.size());
        return 1;
    }
    if (map_test_hz <= 0) {
        ROS_ERROR_STREAM("map_test_hz must be > 0, got: " << map_test_hz);
        return 1;
    }

    ros::Publisher pub_cloud = pnh.advertise<sensor_msgs::PointCloud2>(topic, 10, latch);
    ros::Rate rate(map_test_hz);

    ROS_INFO_STREAM("patchworkpp_pcd_player");
    ROS_INFO_STREAM("pcd_dir   : " << pcd_dir);
    ROS_INFO_STREAM("topic     : " << topic);
    ROS_INFO_STREAM("frame_id  : " << frame_id);
    ROS_INFO_STREAM("hz        : " << map_test_hz);
    ROS_INFO_STREAM("start_idx : " << start_idx);
    ROS_INFO_STREAM("stop mode : " << (step_mode ? "true" : "false"));
    ROS_INFO_STREAM("total pcd : " << pcd_files.size());

    for (int i = start_idx; i < static_cast<int>(pcd_files.size()) && ros::ok(); ++i) {
        if (step_mode) {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        pcl::PCLPointCloud2 pcl_cloud;
        if (pcl::io::loadPCDFile(pcd_files[i], pcl_cloud) < 0) {
            ROS_ERROR_STREAM("Failed to load PCD: " << pcd_files[i]);
            continue;
        }

        sensor_msgs::PointCloud2 cloud_msg;
        pcl_conversions::fromPCL(pcl_cloud, cloud_msg);
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = ros::Time::now();
        const int seq = numericStem(pcd_files[i]);
        cloud_msg.header.seq = (seq >= 0) ? static_cast<uint32_t>(seq) : static_cast<uint32_t>(i);

        ROS_INFO_STREAM("Publishing [" << i << "] " << pcd_files[i]);
        pub_cloud.publish(cloud_msg);
        ros::spinOnce();
        if (!step_mode) rate.sleep();
    }

    ROS_INFO("Finished publishing all PCD frames.");
    return 0;
}
