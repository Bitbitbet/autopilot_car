#include "race_perception.hpp"
#include <iostream>
#include <string>
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    const bool reject=std::string(argv[1])=="reject";
    try {
        RaceConfig cfg;auto result=createRacePerception(cfg);
        if(reject||result) { std::cerr<<"FAIL unexpected backend acceptance\n";return 1; }
        std::cout<<"PASS matching V2 backend contract\n";return 0;
    }catch(const std::exception &e) {
        if(!reject) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
        std::cout<<"PASS incompatible backend rejected: "<<e.what()<<'\n';return 0;
    }
}
