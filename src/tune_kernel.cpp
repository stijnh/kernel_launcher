#include "kernel_launcher/tune_kernel.hpp"

namespace kernel_launcher {

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
            current_time_ += after_event_.seconds_elapsed_since(before_event_);
            state_ = state_tuning;

            if (current_time_ > 1.0) {
                double performance = current_workload_ / current_time_;

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

            uint64_t workload =
                problem_size.x * problem_size.y * problem_size.z;
            current_workload_ += workload;
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
        builder_->compile(current_config_, parameter_types_, *compiler_);
    current_workload_ = 0;
    current_time_ = 0;
}

}  // namespace kernel_launcher