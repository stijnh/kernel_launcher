#include "kernel_launcher/tunable.hpp"

namespace kernel_launcher {

bool RandomStrategy::init(const KernelBuilder& builder, Config& config) {
    _iter = builder.iterate();
    return _iter.next(config);
}

bool RandomStrategy::submit(double, Config& config) {
    return _iter.next(config);
}

bool CachingStrategy::read_cache(
    const KernelBuilder& builder,
    Config& best_config) {
    using nlohmann::json;
    json content;

    std::ifstream stream(_filename.c_str());

    if (stream) {
        content = json::parse(stream);
    } else {
        std::cout << "warning: failed to read: " << _filename << std::endl;
        content = json::object({
            {"kernel", builder.to_json()},
            {"cache", json::array()},
        });
    }

    _cache.clear();
    double best_performance = 1e99;
    bool found_config = false;

    for (const auto& p : content["cache"]) {
        double performance = p["performance"];
        json config = p["config"];

        if (performance < best_performance) {
            best_performance = performance;
            best_config = builder.load_config(config);
            found_config = true;
        }

        _cache[config.dump()] = performance;
    }

    _json = std::move(content);
    return found_config;
}

void CachingStrategy::write_cache(const Config& config, double performance) {
    using nlohmann::json;
    json record;
    record["config"] = config.to_json();
    record["performance"] = performance;

    _json["cache"].push_back(std::move(record));
    std::string output = _json.dump(4);

    std::ofstream stream(
        _filename.c_str(),
        std::ofstream::trunc | std::ofstream::out);
    stream << output;
}

bool CachingStrategy::init(const KernelBuilder& builder, Config& config) {
    if (!_inner->init(builder, config)) {
        return false;
    }

    Config best_config;
    if (read_cache(builder, best_config)) {
        _first_run = true;
        _first_config = std::move(config);
        config = std::move(best_config);
        _current = config.to_json().dump();
        return true;
    }

    return submit_internal(config);
}

bool CachingStrategy::submit(double performance, Config& config) {
    if (_first_run) {
        _first_run = false;
        config = std::move(_first_config);
    } else {
        _cache[_current] = performance;
        write_cache(config, performance);
    }

    if (!_inner->submit(performance, config)) {
        return false;
    }

    return submit_internal(config);
}

bool CachingStrategy::submit_internal(Config& config) {
    while (true) {
        _current = config.to_json().dump();

        auto it = _cache.find(_current);
        if (it == _cache.end()) {
            return true;
        }

        if (!_inner->submit(it->second, config)) {
            return false;
        }
    }
}

void RawTuneKernel::launch(
    cudaStream_t stream,
    dim3 problem_size,
    void** args) {
    if (_state == state_tuning) {
        _after_event.synchronize();
        _current_time += _after_event.elapsed_since(_before_event);

        if (_current_time > 1000.0) {
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