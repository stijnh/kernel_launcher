#include "kernel_launcher/module.hpp"

namespace kernel_launcher {

static std::string cu_error_message(CUresult err) {
    const char* name = "???";
    const char* description = "???";
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &description);

    char buf[1024];
    snprintf(buf, sizeof buf, "CUDA error: %s (%s)", name, description);
    return buf;
}

CuException::CuException(CUresult err) :
    std::runtime_error(cu_error_message(err)),
    _err(err) {};

}  // namespace kernel_launcher