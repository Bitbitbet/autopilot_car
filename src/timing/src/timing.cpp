#include "timing.hpp"
#include <chrono>
#include <cstring>

uint64_t global_microsec() {
    static Clock::time_point start_tp = Clock::now();
    return microsec(start_tp, Clock::now());
}

void LoopTiming::begin() { _start = _marker = Clock::now(); }

void LoopTiming::lap(const char *stage) {
    const auto now = Clock::now();
    if (Stage *found = slot(stage))
        found->sum += microsec(_marker, now);
    _marker = now; // 名字太多时这段就落到 other 里
}

void LoopTiming::end() {
    const auto now = Clock::now();
    _last_loop = microsec(_start, now);
    _total += _last_loop;
    ++_loop_count;
    if (_total < INTERVAL_MS * 1000 && _loop_count < MAX_LOOP_COUNT)
        return;
    report();
    reset();
}

void LoopTiming::reset() {
    _size = 0;
    _total = 0;
    _loop_count = 0;
    // Intentional: don't reset last_loop
}

LoopTiming::Stage *LoopTiming::slot(const char *stage) {
    for (size_t i = 0; i < _size; ++i)
        if (strcmp(_stages[i].name, stage) == 0)
            return &_stages[i];
    if (!stage || _size >= MAX_STAGES)
        return nullptr;
    _stages[_size] = {stage, 0};
    return &_stages[_size++];
}

void LoopTiming::report() {
    printf("[Time] loop=%.1fms", _total / 1000.0 / _loop_count);
    uint64_t sum = 0;
    for (size_t i = 0; i < _size; ++i) {
        printf(" %s=%.1f", _stages[i].name,
               _stages[i].sum / 1000.0 / _loop_count);
        sum += _stages[i].sum;
    }
    printf(" other=%.1f\n", (_total - sum) / 1000.0 / _loop_count);
    fflush(stdout);
}
