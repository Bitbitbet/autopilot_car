#pragma once
#include <exception>
#include <mutex>
#include <optional>
#include <utility>

// 推理和结果拷贝在锁外进行，只交换已完成的快照。
template <typename T> class LatestResult {
    std::mutex mutex;
    std::optional<T> pending;
    std::exception_ptr error;
  public:
    void publish(T result) {
        std::lock_guard lock(mutex);
        pending = std::move(result);
    }
    void fail(std::exception_ptr failure) {
        std::lock_guard lock(mutex);
        error = failure;
    }
    bool take(T &result) {
        std::lock_guard lock(mutex);
        if (error)
            std::rethrow_exception(error);
        if (!pending)
            return false;
        result = std::move(*pending);
        pending.reset();
        return true;
    }
};
