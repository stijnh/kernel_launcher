#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

static const char* demangle_type_info(const std::type_info& type) {
    static std::mutex lock = {};
    static std::unordered_map<std::type_index, std::string> demangled_names =
        {};

    std::lock_guard<std::mutex> guard(lock);
    auto it = demangled_names.find(type);

    if (it != demangled_names.end()) {
        return it->second.c_str();
    }

    const char* mangled_name = type.name();
    int status = ~0;
    // TOOD: look into how portable this solution is on different platforms :-/
    char* undecorated_name =
        abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

    // status == 0: OK
    // status == -1: memory allocation failure
    // status == -2: name is invalid
    // status == -3: one of the other arguments is invalid
    if (status != 0) {
        throw std::runtime_error(
            std::string("__cxa_demangle failed for ") + mangled_name);
    }

    auto result = demangled_names.insert({type, undecorated_name});
    free(undecorated_name);
    return result.first->second.c_str();
}

const char* Type::name() const {
    return demangle_type_info(inner_);
}
}  // namespace kernel_launcher