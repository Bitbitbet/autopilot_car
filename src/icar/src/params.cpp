#include "params.hpp"

using std::cerr;
using std::endl;
using std::make_shared;
using std::string;

Params::Params() {
    // 加载本地json配置文件
    string path = "../res/config.json";
    std::ifstream fileStr(path);
    if (!fileStr.good()) {
        cerr << "Error: Params file path:[" << path << "] not find !!!" << endl;
        exit(-1);
    }
    nlohmann::json configs;
    fileStr >> configs;
    try {
        config = configs.get<Config>();
    } catch (const nlohmann::detail::exception &e) {
        cerr << "Json Params Parse failed :" << e.what() << endl;
        exit(-1);
    }

    mode = FsmMode::normal;                    // 初始化控制模式
    modeLast = FsmMode::normal;                // 初始化控制模式
    track = make_shared<Track>();              // 赛道线处理
    track->rowCutUp = config.rowCutUp;         // 图像顶部切行（前瞻距离）
    track->rowCutBottom = config.rowCutBottom; // 图像底部切行（盲区距离）
};

Params::~Params() {};