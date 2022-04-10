#include "kernel_launcher/value.hpp"

#include <mutex>
#include <unordered_set>

namespace kernel_launcher {
const std::string& intern_string(const std::string& input) {
    static std::mutex lock;
    static std::unordered_map<std::string, std::unique_ptr<std::string>> table;

    std::lock_guard<std::mutex> guard(lock);

    auto it = table.find(input);
    if (it == table.end()) {
        auto result = table.insert(
            std::make_pair(input, std::make_unique<std::string>(input)));
        it = result.first;
    }

    return *(it->second.get());
}
}  // namespace kernel_launcher