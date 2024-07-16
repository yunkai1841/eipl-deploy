#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() {
        reset();
    }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    float getElapsed() {
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(end - start_).count();
    }

    void printElapsed() {
        std::cout << "Time elapsed: " << getElapsed() << " ms" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

class TimerScope {
public:
    TimerScope(const std::string& name) : name_(name) {
        timer_.reset();
    }

    ~TimerScope() {
        std::cout << name_ << " time elapsed: " << timer_.getElapsed() << " ms" << std::endl;
    }

private:
    std::string name_;
    Timer timer_;
};

#endif // __TIMER_H__