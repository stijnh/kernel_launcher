#include "kernel_launcher/module.hpp"

namespace kernel_launcher {

static std::string cu_error_message(
    CUresult err,
    const char* expression,
    const char* filename,
    int line) {
    const char* name = "???";
    const char* description = "???";
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &description);

    char buf[1024];
    snprintf(
        buf,
        sizeof buf,
        "CUDA error: %s (%s) at %s:%d (%s)",
        name,
        description,
        filename,
        line,
        expression);

    return buf;
}

CuException::CuException(
    CUresult err,
    const char* message,
    const char* filename,
    int line) :
    std::runtime_error(cu_error_message(err, message, filename, line)),
    err_(err) {};

}  // namespace kernel_launcher