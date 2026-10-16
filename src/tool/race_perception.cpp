#include "race_perception.hpp"
#include "detection.hpp"
#include <filesystem>
#include <fstream>

extern "C" RacePerception *raceCreatePerception(const RaceConfig *input) {
    const RaceConfig &config=*input;
    namespace fs=std::filesystem;
    const fs::path path=config.modelDirectory;
    // 旧后端将路径传给 NMS 编译命令；入口限制为普通相对/绝对路径，避免 shell 元字符。
    if(path.string().find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-")!=std::string::npos)
        throw std::invalid_argument("Model path contains characters unsupported by legacy NMS builder");
    for(const char *name:{"post.onnx","io_paddle.json","label_list.txt","nms.tar",
                          "deploy_paddle.so","deploy_paddle.params"})
        if(!fs::is_regular_file(path/name)) throw std::runtime_error("Missing model file: "+(path/name).string());
    // 原检测器的两个 PPNC 配置有固定路径，先核对，避免加载另一套权重。
    for(const char *name:{"../res/models/config_ppncnna.json","../res/models/config_ppncnms.json"}) {
        std::ifstream in(name); nlohmann::json j; if(!in) throw std::runtime_error(std::string("Missing ")+name);
        in>>j;
        if(fs::weakly_canonical(j.at("model_dir").get<std::string>())!=fs::weakly_canonical(path))
            throw std::runtime_error(std::string("Model/config path mismatch: ")+name);
    }
    const std::vector<std::string> expected={"cone","person","busy","limit","unlimit",
        "stop","flow","park","gate","cross","fork","left","choice","yleft","yright"};
    std::ifstream labels(path/"label_list.txt"); std::vector<std::string> actual;
    for(std::string line;std::getline(labels,line);) {
        if(!line.empty()&&line.back()=='\r') line.pop_back();
        actual.push_back(line);
    }
    if(actual!=expected) throw std::runtime_error("Model label order differs from race label contract");
    // 模型初始化及其旧版 NMS 编译发生在打开电机串口之前。
    auto detector=std::make_shared<Detection>(path.string(),config.confidence);
    return new RacePerception([detector](const cv::Mat &frame) {
        detector->inference(frame); return detector->results;
    });
}
