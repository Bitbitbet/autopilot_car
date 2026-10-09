# Hardware-free regression checks

These tests never open a real motor controller or start the `icar` application.

For the serial protocol and cleanup test, use a native Linux compiler. The mock
header must precede the real LibSerial include directory:

```sh
g++ -std=c++20 -pthread -Itests/mock -Isrc/control/include \
    tests/control_regression.cpp src/control/src/control.cpp \
    -o /tmp/control_regression
timeout 5 /tmp/control_regression
```

For the real OpenCV track and lost-track checks, enable
`-DCAR_BUILD_REGRESSION_TESTS=ON` in your normal CMake configuration, then build
`track_regression` and `stop_regression`. Run them on the matching target system
or under an ARM64 emulator with the matching sysroot. Assertions and checked
OpenCV access stay enabled in these tests even for Release builds.

- `control_regression`: payload, checksum, deterministic padding; failed open;
  failed writes; disconnect stop; rejection of later drive commands after a stop
  request; bounded join when there is no incoming serial data.
- `track_regression`: first/last row bounds; excessive research row; all-black
  images; invalid image dimensions; worker termination after shutdown.
- `stop_regression`: after 21 consecutive lost-track frames, request shutdown
  without calling `exit`; give zero speed priority; the alternate outline check
  also requests cleanup instead of directly terminating.
- `joystick_regression`: real FIFO and `poll()` with mocked libevdev and serial;
  disconnect stop, rejection of stale drive commands, SIGINT while waiting,
  camera exception, write exception, and joined threads. Links libudev but never
  opens a physical input or serial device.
- `result_regression`: a stalled inference producer cannot block the consumer;
  completed snapshots replace older pending results; model exceptions propagate.

The last two tests are also CMake targets enabled by
`CAR_BUILD_REGRESSION_TESTS`. To run the result test with a native compiler:

```sh
g++ -std=c++20 -pthread -UNDEBUG -Isrc/icar/src \
    tests/result_regression.cpp -o /tmp/result_regression
timeout 5 /tmp/result_regression
```

The existing extra serial byte is retained and initialized to zero. Actual
lower-controller compatibility and motor stopping require separate hardware
verification. These tests cannot provide a motor acknowledgement.
