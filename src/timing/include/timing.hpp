#pragma once

#include <chrono>

using Clock = std::chrono::steady_clock;

inline uint64_t microsec(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
        .count();
}

uint64_t global_microsec();

/**
 * @brief 主循环分段计时
 *
 * 每轮开头 begin()，每做完一段 lap("名字")，结尾 end()；end() 每满 2 秒
 * （或 60 帧）打印一行各段平均耗时（毫秒）。名字传字符串字面量，
 * 内部按内容比较、只存指针，不要传临时字符串。
 *
 *   LoopTiming timing;
 *   while (running) {
 *       timing.begin();
 *       grab();      timing.lap("read");
 *       process();   timing.lap("process");
 *       timing.end();    // [Time] loop=12.3ms read=4.1 process=7.8
 * other=0.4
 *   }
 */
class LoopTiming {
  public:
    static constexpr int MAX_STAGES = 16;

    void begin();

    void lap(const char *stage);

    void end();

    void reset();

    uint64_t last_loop() const { return _last_loop; }

  private:
    static constexpr uint64_t INTERVAL_MS = 2000; // 至少统计这么久
    static constexpr size_t MAX_LOOP_COUNT = 60;  // 或者这么多帧

    struct Stage {
        const char *name = nullptr;
        uint64_t sum; // in microseconds
    };

    decltype(std::chrono::steady_clock::now()) _start, _marker;

    Stage _stages[MAX_STAGES]{};
    size_t _size = 0;

    uint64_t _total = 0; // in microseconds
    size_t _loop_count = 0;

    uint64_t _last_loop = 0;

    Stage *slot(const char *stage);

    void report();
};
