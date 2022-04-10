#include "kernel_launcher/cache.hpp"

#include <unistd.h>

namespace kernel_launcher {

using nlohmann::json;

#define HEADER_MAGIC   ("kernel_launcher")
#define HEADER_VERSION ("0.1")

static std::string current_date() {
    using std::chrono::system_clock;
    std::time_t t = system_clock::to_time_t(system_clock::now());
    std::stringstream oss;
    oss << std::put_time(std::localtime(&t), "%FT%T%z");
    return oss.str();
}

static std::string current_device_name() {
    char name[1024] = {0};
    CUdevice device;
    KERNEL_LAUNCHER_ASSERT(cuCtxGetDevice(&device));
    KERNEL_LAUNCHER_ASSERT(cuDeviceGetName(name, sizeof name, device));
    return name;
}

static json create_header(
    const KernelBuilder& builder,
    const std::vector<TunableParam>& params) {
    std::vector<json> parameters;
    for (const auto& param : params) {
        std::vector<json> values;

        for (const auto& value : param.values()) {
            values.push_back(value.to_json());
        }

        parameters.push_back(
            {{"name", param.name()},
             {"type", param.type().name()},
             {"values", values}});
    }

    char hostname[1024] = {0};
    gethostname(hostname, sizeof hostname);

    int cuda_driver_version;
    KERNEL_LAUNCHER_ASSERT(cuDriverGetVersion(&cuda_driver_version));

    return {
        {"parameters", parameters},
        {"device", current_device_name()},
        {"kernel_name", builder.kernel_name()},
        {"kernel_source", builder.kernel_source().file_name()},
        {"date", current_date()},
        {"version", HEADER_VERSION},
        {"magic", HEADER_MAGIC},
        {"cuda_driver", cuda_driver_version},
        {"hostname", hostname}};
}

static void assert_header_correct(
    const std::string& filename,
    const KernelBuilder& builder,
    const std::vector<TunableParam>& params,
    const json& header) {
    if (header["magic"] != HEADER_MAGIC) {
        throw std::runtime_error(
            "error while opening " + filename
            + ": invalid file format or file has been corrupted");
    }

    if (header["version"] != HEADER_VERSION) {
        throw std::runtime_error(
            "error while opening " + filename + ": invalid version number");
    }

    {
        std::string gotten = header["kernel_name"];
        std::string expected = builder.kernel_name();
        if (expected != gotten) {
            throw std::runtime_error(
                "error while opening " + filename
                + ": results have been tuned for kernel '" + gotten
                + "', but current kernel is '" + expected + "'");
        }
    }

    {
        std::string gotten = header["device"];
        std::string expected = current_device_name();
        if (expected != gotten) {
            throw std::runtime_error(
                "error while opening " + filename
                + ": results have been tuned for device '" + gotten
                + "', but current device is '" + expected + "'");
        }
    }

    bool parameters_valid = true;

    if (params.size() == header["parameters"].size()) {
        size_t i = 0;

        for (const auto& p : header["parameters"]) {
            parameters_valid &= p["name"] == params[i++].name();
        }
    } else {
        parameters_valid = false;
    }

    if (!parameters_valid) {
        throw std::runtime_error(
            "error while opening " + filename
            + ": results have been tuned for different parameters");
    }
}

bool TuningCache::initialize(
    std::string filename,
    const KernelBuilder& builder,
    Config& best_config) {
    filename_ = std::move(filename);
    initialized_ = true;
    parameters_.clear();
    cache_.clear();

    parameters_ = builder.parameters();

    std::sort(parameters_.begin(), parameters_.end(), [](auto a, auto b) {
        return a.name() < b.name();
    });

    std::ifstream stream(filename_.c_str());

    if (!stream) {
        std::ofstream ostream(filename_.c_str());
        ostream << create_header(builder, parameters_);
        return false;
    }

    bool seen_header = false;
    json best_record;
    double best_performance = -std::numeric_limits<double>::infinity();

    for (std::string line; getline(stream, line);) {
        // line.trim(); // TODO: maybe trim string?
        if (line.empty())
            continue;

        json record = json::parse(line);

        if (!seen_header) {
            seen_header = true;
            assert_header_correct(filename_, builder, parameters_, record);
            continue;
        }

        double performance = record["performance"];
        cache_[record["key"]] = performance;

        if (performance > best_performance) {
            best_record = record;
            best_performance = performance;
        }
    }

    if (!best_record.is_null()) {
        best_config = builder.load_config(best_record["config"]);
        return true;
    } else {
        return false;
    }
}

static std::string
config_to_key(const Config& config, const std::vector<TunableParam>& params) {
    std::stringstream output;
    bool is_first = true;

    for (const auto& p : params) {
        if (is_first) {
            is_first = false;
        } else {
            output << "|";
        }

        output << config[p].to_string();
    }

    return output.str();
}

void TuningCache::append(const Config& config, double performance) {
    KERNEL_LAUNCHER_ASSERT(initialized_);
    std::string key = config_to_key(config, parameters_);
    cache_[key] = performance;

    json record = {
        {"key", std::move(key)},
        {"config", config.to_json()},
        {"date", current_date()},
        {"performance", performance}};

    // TODO: Maybe check if file has changed in the meantime somehow?
    std::ofstream stream(filename_, std::ofstream::app);
    stream << "\n" << record;
}

bool TuningCache::find(const Config& config, double& answer) const {
    std::string key = config_to_key(config, parameters_);
    auto it = cache_.find(key);

    if (it == cache_.end()) {
        return false;
    }

    answer = it->second;
    return true;
}
}  // namespace kernel_launcher