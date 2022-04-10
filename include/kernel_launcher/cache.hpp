#pragma once

#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

struct TuningCache {
    TuningCache() {
        //
    }

    bool initialize(
        std::string filename,
        const KernelBuilder& builder,
        Config& best_config);
    void append(const Config& config, double performance);
    bool find(const Config& config, double& performance) const;

  private:
    std::string filename_;
    bool initialized_ = false;
    std::unordered_map<std::string, double> cache_ {};
    std::vector<TunableParam> parameters_ {};
};
}  // namespace kernel_launcher