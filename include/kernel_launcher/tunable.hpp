#pragma once

#include <ios>

#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

struct TuningStrategy {
    virtual bool init(const KernelBuilder& builder, Config& config) = 0;
    virtual bool submit(double performance, Config& config) = 0;
};

struct RandomStrategy: TuningStrategy {
    virtual bool init(const KernelBuilder& builder, Config& config) override {
        _iter = builder.iterate();
        return _iter.next(config);
    }

    virtual bool submit(double, Config& config) override {
        return _iter.next(config);
    }

  private:
    ConfigIterator _iter;
};

template<typename T>
struct CachingStrategy: TuningStrategy {
    CachingStrategy(std::string filename, T inner = {}) :
        _inner(std::move(inner)),
        _filename(std::move(filename)) {
        //
    }

    bool read_cache(const KernelBuilder& builder, Config& best_config) {
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

    void write_cache(const Config& config, double performance) {
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

    virtual bool init(const KernelBuilder& builder, Config& config) override {
        if (!_inner.init(builder, config)) {
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

    virtual bool submit(double performance, Config& config) override {
        if (_first_run) {
            _first_run = false;
            config = std::move(_first_config);
        } else {
            _cache[_current] = performance;
            write_cache(config, performance);
        }

        if (!_inner.submit(performance, config)) {
            return false;
        }

        return submit_internal(config);
    }

  private:
    bool submit_internal(Config& config) {
        while (true) {
            _current = config.to_json().dump();

            auto it = _cache.find(_current);
            if (it == _cache.end()) {
                return true;
            }

            if (!_inner.submit(it->second, config)) {
                return false;
            }
        }
    }

  private:
    T _inner;

    std::string _filename;
    nlohmann::json _json;

    bool _first_run;
    Config _first_config;
    std::string _current;
    std::unordered_map<std::string, double> _cache;
};

struct RawTuneKernel {
    RawTuneKernel() : _state(state_uninitialized) {
        //
    }

    RawTuneKernel(
        KernelBuilder builder,
        std::vector<Type> parameter_types,
        std::unique_ptr<Compiler> compiler = {},
        std::unique_ptr<TuningStrategy> strategy = {}) :
        _state(state_init),
        _builder(std::make_unique<KernelBuilder>(std::move(builder))),
        _strategy(std::move(strategy)),
        _compiler(std::move(compiler)),
        _parameter_types(std::move(parameter_types)) {
        if (!_strategy) {
            _strategy = std::make_unique<RandomStrategy>();
        }

        if (!_compiler) {
            _compiler = std::make_unique<NvrtcCompiler>();
        }

        next_configuration();
    }

    void launch(cudaStream_t stream, dim3 problem_size, void** args) {
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

  private:
    void next_configuration() {
        if (_state == state_init) {
            if (!_strategy->init(*_builder, _current_config)) {
                throw std::runtime_error(
                    "search strategy failed to initialize");
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

    enum {
        state_uninitialized,
        state_init,
        state_tuning,
        state_compiling,
        state_finished,
    } _state;

    std::unique_ptr<KernelBuilder> _builder;
    std::unique_ptr<TuningStrategy> _strategy;
    std::unique_ptr<Compiler> _compiler;
    std::vector<Type> _parameter_types;

    CudaEvent _before_event;
    CudaEvent _after_event;

    double _best_performance = 1e9;
    RawKernel _best_kernel;

    Config _current_config;
    double _current_time = 0;
    uint64_t _current_workload = 0;
    RawKernel _current_kernel;
    bool _first_run;
};

template<typename... Args>
struct TuneKernel {
    using instance_type = KernelInstantiation<RawTuneKernel, Args...>;

    TuneKernel() {}

    TuneKernel(
        KernelBuilder builder,
        std::unique_ptr<TuningStrategy> strategy = {},
        std::unique_ptr<Compiler> compiler = {}) :
        _kernel(
            std::move(builder),
            {type_of<Args>()...},
            std::move(compiler),
            std::move(strategy)) {}

    instance_type instantiate(cudaStream_t stream, dim3 problem_size) {
        return instance_type(stream, problem_size, _kernel);
    }

    instance_type operator()(cudaStream_t stream, dim3 problem_size) {
        return instantiate(stream, problem_size);
    }

    instance_type operator()(dim3 problem_size) {
        return instantiate(nullptr, problem_size);
    }

    instance_type operator()(
        cudaStream_t stream,
        uint32_t problem_x,
        uint32_t problem_y,
        uint32_t problem_z = 1) {
        return instantiate(stream, dim3(problem_x, problem_y, problem_z));
    }

    instance_type
    operator()(uint32_t problem_x, uint32_t problem_y, uint32_t problem_z = 1) {
        return instantiate(nullptr, dim3(problem_x, problem_y, problem_z));
    }

    RawTuneKernel _kernel;
};
}  // namespace kernel_launcher
