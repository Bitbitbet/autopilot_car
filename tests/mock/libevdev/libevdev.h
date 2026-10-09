#pragma once
#include <cerrno>
#include <linux/input.h>
#include <unistd.h>
// Test double: use a FIFO to exercise the real poll()/disconnect path.
struct libevdev { int fd; };
inline constexpr int LIBEVDEV_READ_FLAG_NORMAL = 0;
inline constexpr int LIBEVDEV_READ_STATUS_SUCCESS = 0;
inline constexpr int LIBEVDEV_READ_STATUS_SYNC = 1;
inline int libevdev_new_from_fd(int fd, libevdev **dev) {
    *dev = new libevdev{fd};
    return 0;
}
inline void libevdev_free(libevdev *dev) { delete dev; }
inline const input_absinfo *libevdev_get_abs_info(libevdev *, unsigned int) {
    static const input_absinfo range{0, 0, 255, 0, 0, 0};
    return &range;
}
inline int libevdev_next_event(libevdev *dev, unsigned int, input_event *event) {
    auto size = read(dev->fd, event, sizeof(*event));
    if (size == sizeof(*event)) return LIBEVDEV_READ_STATUS_SUCCESS;
    if (size < 0) return -errno;
    return -ENODEV;
}
