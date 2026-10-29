#pragma once

#include <cerrno>
#include <termios.h>
#include <unistd.h>

/**
 * @brief 终端按键：关掉行缓冲和回显，非阻塞读单个字符
 *
 * 用于不依赖图像窗口的操作（headless 时也能按 a/空格/q）。
 * stdin 不是终端时自动失效，poll() 恒返回 -1。
 */
class TerminalKeys {
    termios saved{};
    bool active = false;

  public:
    TerminalKeys() {
        if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &saved) != 0)
            return;
        termios raw = saved;
        raw.c_lflag &= ~(ICANON | ECHO); // Ctrl-C 等信号键保留
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0)
            active = true;
    }
    ~TerminalKeys() {
        if (active)
            tcsetattr(STDIN_FILENO, TCSANOW, &saved);
    }
    TerminalKeys(const TerminalKeys &) = delete;
    TerminalKeys &operator=(const TerminalKeys &) = delete;

    /**
     * @brief 取一个按键，没有就返回 -1
     */
    int poll() {
        if (!active)
            return -1;
        unsigned char c = 0;
        const ssize_t n = ::read(STDIN_FILENO, &c, 1);
        return n == 1 ? int(c) : -1;
    }
};
