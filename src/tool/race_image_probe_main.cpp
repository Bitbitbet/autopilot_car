#include <cstddef>
#include <iostream>
extern "C" int raceImageProbe(const char*,const char*,char*,size_t) noexcept;

int main(int argc,char **argv) {
    if(argc!=3) { std::cerr<<"Saved image diagnostic: manifest.json output.jsonl; no camera or vehicle control.\n"; return 2; }
    char error[4096]={};
    const int rc=raceImageProbe(argv[1],argv[2],error,sizeof(error));
    if(rc) std::cerr<<error<<'\n';
    return rc;
}
