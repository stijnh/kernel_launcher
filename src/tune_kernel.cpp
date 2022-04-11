#include "kernel_launcher/tune_kernel.hpp"

namespace kernel_launcher {

void Aggregator::reset() {
    records_.clear();
}

void Aggregator::add(dim3 problem_size, double time) {
    records_.push_back({problem_size, time});
}

bool Aggregator::collect(double& performance) {
    double total_time = 0.0;
    double total_workload = 0.0;

    for (auto p : records_) {
        total_workload += p.first.x * p.first.y * p.first.z;
        total_time += p.second;
    }

    // We need more runs if we have not reached max_evals_ or max_seconds_
    if (records_.size() < max_evals_ && total_time < max_seconds_) {
        return false;
    }

    performance = total_workload / total_time;
    return true;
}

void RawTuneKernel::launch(
    cudaStream_t stream,
    dim3 problem_size,
    void** args) {
    while (1) {
        // Finished tuning, just launch the best kernel
        if (state_ == state_finished) {
            best_kernel_.launch(stream, problem_size, args);
            return;
        }

        // Measure performance of kernel
        else if (state_ == state_measuring) {
            after_event_.synchronize();
            double time = after_event_.seconds_elapsed_since(before_event_);
            aggregator_.add(current_problem_, time);
            state_ = state_tuning;

            double performance;
            if (aggregator_.collect(performance)) {
                if (performance > best_performance_) {
                    best_kernel_ = std::exchange(current_kernel_, {});
                    best_performance_ = performance;
                }

                if (!strategy_.submit(performance, current_config_)) {
                    state_ = state_finished;
                    builder_.reset();
                    strategy_.reset();
                    compiler_.reset();
                    continue;
                }

                next_configuration();
            }
        }

        // Launch tuning run
        else if (state_ == state_tuning) {
            before_event_.record(stream);
            current_kernel_.launch(stream, problem_size, args);
            after_event_.record(stream);

            current_problem_ = problem_size;
            state_ = state_measuring;
            return;
        }

        // Waiting for compilation of kernel
        else if (state_ == state_compiling) {
            after_event_.synchronize();

            if (current_kernel_.ready()) {
                state_ = state_tuning;
            } else if (best_kernel_.ready()) {
                best_kernel_.launch(stream, problem_size, args);
                after_event_.record(stream);
                return;
            } else {
                current_kernel_.wait_ready();
            }
        }

        // Invalid state?
        else {
            throw std::runtime_error("kernel has not been initialized");
        }
    }
}

void RawTuneKernel::next_configuration() {
    state_ = state_compiling;
    current_kernel_ =
        builder_->compile(current_config_, parameter_types_, compiler_);
    aggregator_.reset();
}

}  // namespace kernel_launcher