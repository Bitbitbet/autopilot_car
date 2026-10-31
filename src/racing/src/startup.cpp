#include "startup.hpp"
#include "startup_script.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

int raceVehicleMain(int,char**);
namespace fs=std::filesystem;
namespace {
bool modeValid(const std::string &mode){
    static constexpr std::array modes{"preview","drive","calibrate","bypass","parking-preview","parking-drive","reverse-check","bypass-virtual","bypass-virtual-preview","mission","mission-preview","mission-lab","mission-lab-preview"};
    for(const char *candidate:modes)if(mode==candidate)return true;
    return false;
}
int prepare(const std::string &mode,const std::string &profile,const std::string &surface,const std::string &prefix,const std::string &binary){
    std::cout.flush();std::cerr.flush();
    const pid_t child=fork();
    if(child<0)throw std::runtime_error("Cannot create startup preparation process");
    if(child==0){
        execlp("python3","python3","-c",raceStartupScript,mode.c_str(),profile.c_str(),prefix.c_str(),surface.c_str(),binary.c_str(),static_cast<char*>(nullptr));
        const char message[]="Startup requires the vehicle's existing python3.\n";
        const auto ignored=write(STDERR_FILENO,message,sizeof(message)-1);(void)ignored;
        _exit(127);
    }
    int status=0;
    while(waitpid(child,&status,0)<0){if(errno!=EINTR)throw std::runtime_error("Cannot wait for startup preparation");}
    return WIFEXITED(status)?WEXITSTATUS(status):128+(WIFSIGNALED(status)?WTERMSIG(status):0);
}
}

int raceLaunchMain(int argc,char **argv){
    try{
        const std::string first=argc>1?argv[1]:"";
        if(first=="--help"){
            std::cout<<"Automatic preparation: ./racing [drive|mission|parking-drive|bypass-virtual|preview|mission-preview|mission-lab] [competition|lane|signal|limit] [white|black|config]\n"
                <<"No arguments: prepare drive competition using configured road surface; press a to start.\n"
                <<"--prepare-only [mode] [profile] [surface]: validate and write launch record; no camera, AI or motor initialization.\n"
                <<"Full low-level --config/--race-config options remain available for prepared configurations.\n";
            return raceVehicleMain(argc,argv);
        }
        const bool only=first=="--prepare-only";
        const bool launch=first=="--launch";
        // Preserve complete low-level argument lists; shorthand and no-argument
        // starts always use the single shared preparation implementation.
        if(!first.empty()&&first.rfind("--",0)==0&&!only&&!launch)return raceVehicleMain(argc,argv);
        const int start=only||launch?2:1;
        if(argc-start>3)throw std::invalid_argument("Use racing [mode] [profile] [white|black|config]");
        const std::string mode=argc>start?argv[start]:"drive";
        const std::string profile=argc>start+1?argv[start+1]:"competition";
        const std::string surface=argc>start+2?argv[start+2]:"config";
        if(!modeValid(mode)||(profile!="competition"&&profile!="lane"&&profile!="signal"&&profile!="limit")||
            (surface!="config"&&surface!="white"&&surface!="black"))throw std::invalid_argument("Unknown startup mode/profile/surface");
        const auto executable=fs::read_symlink("/proc/self/exe");
        fs::current_path(executable.parent_path());
        fs::create_directories("run-records");
        const auto clock=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm local{};localtime_r(&clock,&local);char stamp[32];std::strftime(stamp,sizeof(stamp),"%Y%m%d-%H%M%S",&local);
        const std::string prefix=std::string("./run-records/")+stamp+'-'+std::to_string(getpid())+'-'+profile;
        const int result=prepare(mode,profile,surface,prefix,executable.filename().string());
        if(result)return result;
        if(only){std::cout<<"[Startup] PREPARED ONLY | no camera, AI or motor initialized\n";return 0;}
        std::ifstream input(prefix+"-launch.json");nlohmann::json record;
        if(!input)throw std::runtime_error("Cannot read prepared launch record");
        input>>record;
        auto arguments=record.at("command").get<std::vector<std::string>>();
        if(arguments.empty())throw std::runtime_error("Prepared command is empty");
        std::vector<char*> pointers;for(auto &argument:arguments)pointers.push_back(argument.data());
        pointers.push_back(nullptr);
        return raceVehicleMain(int(arguments.size()),pointers.data());
    }catch(const std::exception &error){std::cerr<<"[Startup] "<<error.what()<<'\n';return 2;}
}
