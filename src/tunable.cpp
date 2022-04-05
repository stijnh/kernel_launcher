#include "kernel_launcher/tunable.hpp"

#include <unistd.h>

namespace kernel_launcher {

using nlohmann::json;

bool RandomStrategy::init(const KernelBuilder& builder, Config& config) {
    _iter = builder.iterate();
    return _iter.next(config);
}

bool RandomStrategy::submit(double, Config& config) {
    return _iter.next(config);
}

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

#define HEADER_MAGIC   ("kernel_launcher")
#define HEADER_VERSION ("0.1")

static json create_header(
    const KernelBuilder& builder,
    const std::vector<TunableParam>& params) {
    std::vector<json> parameters;
    for (const auto& param : params) {
        std::vector<json> values;

        for (const auto& val : builder.parameters().at(param)) {
            values.push_back(val.to_json());
        }

        parameters.push_back(
            {{"name", param.name()},
             {"type", param.type().name()},
             {"values", values}});
    }

    std::sort(parameters.begin(), parameters.end(), [&](auto a, auto b) {
        return a < b;
    });

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
    const KernelBuilder& builder,
    Config& best_config) {
    _initialized = true;
    _parameters.clear();
    _cache.clear();

    for (const auto& p : builder.parameters()) {
        _parameters.push_back(p.first);
    }

    std::sort(_parameters.begin(), _parameters.end(), [](auto a, auto b) {
        return a.name() < b.name();
    });

    std::ifstream stream(_filename.c_str());

    if (!stream) {
        std::ofstream ostream(_filename.c_str());
        ostream << create_header(builder, _parameters);
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
            assert_header_correct(_filename, builder, _parameters, record);
            continue;
        }

        double performance = record["performance"];
        _cache[record["key"]] = performance;

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
    KERNEL_LAUNCHER_ASSERT(_initialized);
    std::string key = config_to_key(config, _parameters);
    _cache[key] = performance;

    json record = {
        {"key", std::move(key)},
        {"config", config.to_json()},
        {"date", current_date()},
        {"performance", performance}};

    // TODO: Maybe check if file has changed in the meantime somehow?
    std::ofstream stream(_filename, std::ofstream::app);
    stream << "\n" << record;
}

bool TuningCache::find(const Config& config, double& answer) const {
    std::string key = config_to_key(config, _parameters);
    auto it = _cache.find(key);

    if (it == _cache.end()) {
        return false;
    }

    answer = it->second;
    return true;
}

static bool internal_submit(
    const TuningCache& cache,
    TuningStrategy& inner,
    Config& config) {
    double perf;

    while (true) {
        if (!cache.find(config, perf)) {
            return true;
        }

        if (!inner.submit(perf, config)) {
            return false;
        }
    }
}

bool CachingStrategy::init(const KernelBuilder& builder, Config& config) {
    if (!_inner->init(builder, config)) {
        return false;
    }

    Config best_config;
    if (_cache.initialize(builder, best_config)) {
        _first_run = true;
        _first_config = std::move(config);
        config = std::move(best_config);
        return true;
    } else {
        _first_run = false;
    }

    return internal_submit(_cache, *_inner, config);
}

bool CachingStrategy::submit(double performance, Config& config) {
    if (_first_run) {
        _first_run = false;
        config = std::move(_first_config);
    } else {
        _cache.append(config, 1 / performance);

        if (!_inner->submit(performance, config)) {
            return false;
        }
    }

    return internal_submit(_cache, *_inner, config);
}

void RawTuneKernel::launch(
    cudaStream_t stream,
    dim3 problem_size,
    void** args) {
    if (_state == state_tuning) {
        _after_event.synchronize();
        _current_time += _after_event.seconds_elapsed_since(_before_event);

        if (_current_time > 1.0) {
            next_configuration();
        }
    }

    if (_state == state_finished) {
        _best_kernel.launch(stream, problem_size, args);
        return;
    }

    if (_state == state_compiling) {
        if (!_current_kernel.ready() && _best_kernel.ready()) {
            _best_kernel.launch(stream, problem_size, args);
            return;
        }

        _current_kernel.wait_ready();
        _state = state_tuning;
    }

    if (_state != state_tuning) {
        throw std::runtime_error("kernel has not been initialized");
    }

    _before_event.record(stream);
    _current_kernel.launch(stream, problem_size, args);
    _after_event.record(stream);

    uint64_t workload = problem_size.x * problem_size.y * problem_size.z;
    _current_workload += workload;
}

void RawTuneKernel::next_configuration() {
    if (_state == state_init) {
        if (!_strategy->init(*_builder, _current_config)) {
            throw std::runtime_error("search strategy failed to initialize");
        }
    } else if (_state == state_tuning) {
        double performance = _current_time / _current_workload;

        if (performance < _best_performance) {
            _best_kernel = std::exchange(_current_kernel, {});
            _best_performance = performance;
        }

        if (!_strategy->submit(performance, _current_config)) {
            _state = state_finished;
            _builder.reset();
            _strategy.reset();
            _compiler.reset();
            return;
        }
    }

    _state = state_compiling;
    _current_kernel =
        _builder->compile(_current_config, _parameter_types, *_compiler);
    _current_workload = 0;
    _current_time = 0;
}

}  // namespace kernel_launcher