#pragma once

#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

struct TuningCache {
    bool open(
        std::string filename,
        const KernelBuilder& builder,
        Config& best_config);
    void append(const Config& config, double performance);
    bool find(const Config& config, double& performance) const;

  private:
    std::string filename_ = {};
    std::unordered_map<std::string, double> cache_ {};
    std::vector<TunableParam> parameters_ {};
};
}  // namespace kernel_launcher

#if KERNEL_LAUNCHER_HEADERONLY
    #include KERNEL_LAUNCHER_IMPL("cache.cpp")
#endif