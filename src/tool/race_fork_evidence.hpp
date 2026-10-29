#pragma once
#include <string>
#include <cstdint>

// Save state transitions and at most two approach frames per second, via the existing
// bounded writer. A full queue never blocks control; a transition is retried.
class RaceForkEvidence {
    std::string savedKey;
    bool savedFork=false;
    double lastTime=-1;
    uint64_t lastSequence=0;
    int count=0;
    static bool fork(const std::string &state,const std::string &reason) {
        return state.rfind("FORK",0)==0 || reason.rfind("FORK",0)==0 ||
            reason.find("DIRECTION")!=std::string::npos || reason=="BRANCH_PATH_LOST";
    }
public:
    int nextIndex() const { return count+1; }
    bool due(const std::string &state,const std::string &reason,double now,uint64_t sequence) const {
        if(!sequence||count>=64)return false;
        const bool current=fork(state,reason);
        if(!current&&!savedFork)return false;
        const std::string key=state+"|"+reason;
        if(key!=savedKey)return true;
        return current&&state!="REVIEW_STOP"&&sequence!=lastSequence&&now-lastTime>=.5;
    }
    void queued(const std::string &state,const std::string &reason,double now,uint64_t sequence) {
        savedKey=state+"|"+reason;savedFork=fork(state,reason);lastTime=now;lastSequence=sequence;++count;
    }
};
