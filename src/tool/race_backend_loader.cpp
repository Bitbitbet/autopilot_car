#include "race_perception.hpp"
#include <dlfcn.h>
#include <filesystem>

// PPNC 库存在加载期设备认证。仅在实车推理模式 dlopen，帮助和回放不碰硬件。
std::unique_ptr<RacePerception> createRacePerception(const RaceConfig &config) {
    const auto library=std::filesystem::read_symlink("/proc/self/exe").parent_path()/"librace_npu_backend.so";
    void *handle=dlopen(library.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!handle) {
        const char *error=dlerror();
        throw std::runtime_error("Cannot load vehicle AI backend: "+std::string(error?error:"unknown"));
    }
    using Create=RacePerception*(*)(const RaceConfig*);
    auto create=reinterpret_cast<Create>(dlsym(handle,"raceCreatePerception"));
    if(!create) throw std::runtime_error("Vehicle AI backend entry point missing");
    // 回调与析构代码属于动态库，句柄保留到进程退出，不能在线程结束前 dlclose。
    return std::unique_ptr<RacePerception>(create(&config));
}
