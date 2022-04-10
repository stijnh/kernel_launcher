#include "kernel_launcher/value.hpp"

#include <mutex>
#include <unordered_set>

namespace kernel_launcher {
const std::string& intern_string(const char* input) {
    auto equal = [](const char* a, const char* b) { return strcmp(a, b) == 0; };
    auto hash = [](const char* v) {
        size_t h = 0;
        for (; *v; v++) {
            h = h * 31 + *v;
        }
        return h;
    };

    static std::mutex lock;
    static std::unordered_map<
        const char*,
        std::unique_ptr<std::string>,
        decltype(hash),
        decltype(equal)>
        table(32, hash, equal);

    std::lock_guard<std::mutex> guard(lock);

    auto it = table.find(input);
    if (it == table.end()) {
        auto value = std::make_unique<std::string>(input);
        auto key = value->c_str();

        auto result = table.insert(std::make_pair(key, std::move(value)));
        it = result.first;
    }

    return *(it->second.get());
}
}  // namespace kernel_launcher