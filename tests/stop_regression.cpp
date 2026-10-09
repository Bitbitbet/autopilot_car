#include "center.hpp"
#include "motion.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

int main() {
    namespace fs = std::filesystem;
    auto previous = fs::current_path();
    auto root = fs::temp_directory_path() / ("icar-stop-test-" + std::to_string(getpid()));
    fs::create_directories(root / "res");
    fs::create_directories(root / "bin");
    std::ofstream(root / "res/config.json") << nlohmann::json(Config{});
    fs::current_path(root / "bin");
    auto params = std::make_shared<Params>();
    Center center;
    for (int i = 0; i < 20; ++i) {
        center.fitting(params);
        assert(!params->quit);
    }
    center.fitting(params);
    assert(params->quit && params->ctrl.stop);
    Motion motion;
    params->ctrl.speed = 1.0;
    motion.speedControl(params);
    assert(params->ctrl.speed == 0.0);
    auto other = std::make_shared<Params>();
    Motion outline;
    for (int i = 0; i < 30 && !other->quit; ++i) outline.outlineCheck(other);
    assert(other->quit && other->ctrl.stop); // requests cleanup instead of exit()
    fs::current_path(previous);
    fs::remove_all(root);
    std::cout << "PASS: lost-track shutdown request and zero-speed priority\n";
}
