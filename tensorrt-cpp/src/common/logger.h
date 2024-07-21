#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <iostream>
#include <NvInfer.h>

#ifndef __DISABLE_LOGGING__
#define LOG_VERBOSE(...) \
    std::cout << "[VERBOSE] " << __VA_ARGS__ << std::endl;
#define LOG_INFO(...) \
    std::cout << "[INFO] " << __VA_ARGS__ << std::endl;
#define LOG_WARNING(...) \
    std::cerr << "[WARNING] " << __VA_ARGS__ << std::endl;
#define LOG_ERROR(...) \
    std::cerr << "[ERROR] " << __VA_ARGS__ << std::endl;
#else
#define LOG_VERBOSE(...)
#define LOG_INFO(...)
#define LOG_WARNING(...)
#define LOG_ERROR(...)
#endif

class Logger : public nvinfer1::ILogger
{
public:
    Logger(const Severity severity = Severity::kWARNING) : mSeverity(severity) {}
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > mSeverity)
        {
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            LOG_ERROR(msg);
            break;
        case Severity::kERROR:
            LOG_ERROR(msg);
            break;
        case Severity::kWARNING:
            LOG_WARNING(msg);
            break;
        case Severity::kINFO:
            LOG_INFO(msg);
            break;
        case Severity::kVERBOSE:
            LOG_VERBOSE(msg);
            break;
        default:
            break;
        }
    }

    Severity getSeverity() const noexcept
    {
        return mSeverity;
    }

    void setSeverity(Severity severity) noexcept
    {
        mSeverity = severity;
    }

private:
    Severity mSeverity;
};

#endif // __LOGGER_H__