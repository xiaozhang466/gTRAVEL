// For disable PCL compile lib, to use PointXYZILID
#define PCL_NO_PRECOMPILE

#include "gtravelp/node.h"
#include "utils/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <dirent.h>
#include <iostream>
#include <limits>
#include <thread>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

namespace {

std::string basenameNoExt(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

bool isDigits(const std::string& s) {
    return !s.empty() &&
           std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

int numericStem(const std::string& path) {
    const std::string stem = basenameNoExt(path);
    if (!isDigits(stem)) return -1;
    return std::stoi(stem);
}

template <typename T>
sensor_msgs::PointCloud2 cloud2msg(const pcl::PointCloud<T>& cloud, const std::string& frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ros;
    pcl::toROSMsg(cloud, cloud_ros);
    cloud_ros.header.frame_id = frame_id;
    return cloud_ros;
}

std::vector<std::string> collectPcdFiles(const std::string& dir_path) {
    std::vector<std::string> files;

    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return files;

    dirent* ent = nullptr;
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

    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
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

int main(int argc, char** argv) {
    ros::init(argc, argv, "Ros-MapTest-Publisher");
    ros::NodeHandle nh;

    std::string node_topic;
    std::string pcd_dir;
    int start_idx;
    int map_test_hz;
    bool step_mode;
    bool wait_subscribers;
    int min_subscribers;
    double startup_delay_sec;

    nh.param<std::string>("/node_topic", node_topic, "/node");
    nh.param<std::string>("/pcd_dir", pcd_dir, "/");
    nh.param<int>("/start_idx", start_idx, 0);
    nh.param<int>("/map_test_hz", map_test_hz, 10);
    nh.param<bool>("/stop", step_mode, false);
    nh.param<bool>("/map_test_wait_subscribers", wait_subscribers, true);
    nh.param<int>("/map_test_min_subscribers", min_subscribers, 1);
    nh.param<double>("/map_test_startup_delay_sec", startup_delay_sec, 2.0);

    std::cout << "\033[1;32mNode topic: " << node_topic << "\033[0m" << std::endl;
    std::cout << "\033[1;32mPCD directory: " << pcd_dir << "\033[0m" << std::endl;

    std::vector<std::string> pcd_files = collectPcdFiles(pcd_dir);
    if (pcd_files.empty()) {
        std::cerr << "\033[1;31mError: no .pcd files found in " << pcd_dir << "\033[0m" << std::endl;
        return 1;
    }

    if (start_idx < 0 || start_idx >= static_cast<int>(pcd_files.size())) {
        std::cerr << "\033[1;31mError: start_idx out of range: " << start_idx
                  << " (total: " << pcd_files.size() << ")\033[0m" << std::endl;
        return 1;
    }

    ros::Publisher node_publisher = nh.advertise<gtravelp::node>(node_topic, 100, true);
    ros::Rate rate(map_test_hz);

    if (wait_subscribers) {
        std::cout << "\033[1;32m[MapTest Publisher] Waiting subscribers on " << node_topic
                  << " (>= " << min_subscribers << ")\033[0m" << std::endl;
        while (ros::ok() && static_cast<int>(node_publisher.getNumSubscribers()) < std::max(1, min_subscribers)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::cout << "\033[1;32m[MapTest Publisher] Subscribers ready: "
                  << node_publisher.getNumSubscribers() << "\033[0m" << std::endl;
    }

    if (startup_delay_sec > 0.0) {
        std::cout << "\033[1;32m[MapTest Publisher] Startup delay: " << startup_delay_sec
                  << " sec\033[0m" << std::endl;
        ros::Duration(startup_delay_sec).sleep();
    }

    std::cout << "\033[1;32m[MapTest Publisher] Total " << pcd_files.size() << " clouds are loaded\033[0m" << std::endl;
    for (int i = start_idx; i < static_cast<int>(pcd_files.size()) && ros::ok(); ++i) {
        if (step_mode) {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        pcl::PointCloud<pcl::PointXYZINormal> input_cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZINormal>(pcd_files[i], input_cloud) < 0) {
            std::cerr << "\033[1;31mFailed to load " << pcd_files[i] << "\033[0m" << std::endl;
            continue;
        }

        pcl::PointCloud<PointXYZILID> output_cloud;
        output_cloud.reserve(input_cloud.size());
        for (const auto& pt : input_cloud.points) {
            PointXYZILID out_pt;
            out_pt.x = pt.x;
            out_pt.y = pt.y;
            out_pt.z = pt.z;
            out_pt.intensity = pt.intensity;
            out_pt.label = 0;
            out_pt.id = 0;
            output_cloud.push_back(out_pt);
        }

        const int seq = numericStem(pcd_files[i]);
        gtravelp::node node;
        node.lidar = cloud2msg(output_cloud);
        node.header = node.lidar.header;
        node.header.seq = (seq >= 0) ? seq : i;

        std::cout << "Publishing " << i << " -> " << pcd_files[i] << std::endl;
        node_publisher.publish(node);
        rate.sleep();
    }

    return 0;
}
