#include "latest_result.hpp"
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

int main() {
    LatestResult<std::vector<int>> mailbox;
    std::promise<void> started, resume;
    auto resumeFuture = resume.get_future();
    std::thread inference([&] {
        started.set_value();
        resumeFuture.wait(); // Deliberately stalled inference.
        mailbox.publish({1, 2, 3});
    });
    started.get_future().wait();
    auto consumer = std::async(std::launch::async, [&] {
        std::vector<int> result{7};
        assert(!mailbox.take(result));
        assert(result == std::vector<int>{7});
    });
    assert(consumer.wait_for(1s) == std::future_status::ready);
    consumer.get();
    resume.set_value();
    inference.join();
    std::vector<int> snapshot;
    assert(mailbox.take(snapshot) && snapshot == (std::vector<int>{1, 2, 3}));
    mailbox.publish({4});
    mailbox.publish({5});
    assert(mailbox.take(snapshot) && snapshot == std::vector<int>{5});
    try { throw std::runtime_error("model failure"); }
    catch (...) { mailbox.fail(std::current_exception()); }
    bool caught = false;
    try { mailbox.take(snapshot); }
    catch (const std::runtime_error &) { caught = true; }
    assert(caught);
    std::cout << "PASS: stalled inference does not block consumer, latest snapshot, exception propagation\n";
}
