#pragma once

#include <random>
#include <unordered_set>
#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

    struct TuningCache {
        TuningCache(std::string filename) : filename_(std::move(filename)) {
            //
        }

        bool initialize(const KernelBuilder& builder, Config& best_config);
        void append(const Config& config, double performance);
        bool find(const Config& config, double& performance) const;

    private:
        bool initialized_ = false;
        std::string filename_;
        std::unordered_map<std::string, double> cache_ {};
        std::vector<TunableParam> parameters_ {};
    };

    struct TuningStrategy {
        virtual bool init(const KernelBuilder& builder, Config& config) = 0;
        virtual bool submit(double performance, Config& config) = 0;
        virtual ~TuningStrategy() = default;
    };

    struct RandomStrategy: TuningStrategy {
        bool init(const KernelBuilder& builder, Config& config) override;
        bool submit(double performance, Config& config) override;

    private:
        ConfigIterator iter_;
    };

    struct HillClimbingStrategy: TuningStrategy {
        HillClimbingStrategy(std::unique_ptr<TuningStrategy> inner): inner_(std::move(inner)) {

        }

        bool init(const KernelBuilder& builder, Config& config) override;
        bool submit(double performance, Config& config) override;

        private:
            void update_best(double performance, const Config& config);

            std::default_random_engine rng_;
            std::vector<bool> attempted_neighbors_;
            std::vector<std::pair<TunableParam, TunableValue>> neighbors_;
            size_t attempted_neighbors_count_;
            ConfigSpace space_;
            std::unique_ptr<TuningStrategy> inner_;
            double best_performance_;
            Config best_config_;
    };

    struct LimitStrategy: TuningStrategy {
        LimitStrategy(uint64_t max_eval, std::unique_ptr<TuningStrategy> inner): max_eval_(max_eval), inner_(std::move(inner)) {

        }

        bool init(const KernelBuilder& builder, Config& config) override;
        bool submit(double performance, Config& config) override;

    private:
        uint64_t curr_eval_;
        uint64_t max_eval_;
        std::unique_ptr<TuningStrategy> inner_;
    };

    struct CachingStrategy: TuningStrategy {
        template<typename T>
        CachingStrategy(std::string filename, T inner = {}) :
        inner_(std::make_unique<std::decay_t<T>>(std::forward<T>(inner))),
        cache_(std::move(filename)) {
            //
        }

        bool init(const KernelBuilder& builder, Config& config) override;
        bool submit(double performance, Config& config) override;

    private:
        std::unique_ptr<TuningStrategy> inner_;
        TuningCache cache_;
        bool first_run_;
        Config first_config_;
    };

}