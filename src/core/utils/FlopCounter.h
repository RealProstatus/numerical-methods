#pragma once

#include <cstdint>
#include <chrono>
#include <iostream>
#include <papi.h>

namespace profiling
{

// Just DTO
struct FlopStats
{
    uint64_t  flop_cnt{};
    double    elapsed_ms{};

    double gflops() const
    {
        if (elapsed_ms > 0.0)
        {
            return static_cast<double>(flop_cnt) * 1e-6 / elapsed_ms;
        }
        return 0.0;
    }
};

class FlopCounter
{
    inline static thread_local uint64_t total_flop_cnt{};
    inline static thread_local bool     enabled{};
    
public:
    static inline void add(uint64_t count) noexcept
    {
        if (enabled)
        {
            total_flop_cnt += count;
        }
    }

    static inline void add_explicit(uint64_t count) noexcept
    {
        total_flop_cnt += count;
    }

    static inline uint64_t get_total_flops() noexcept { return total_flop_cnt; }
    static inline bool is_enabled() noexcept { return enabled; }
    static inline void set_enabled(bool state) noexcept { enabled = state; }

    // Заглушки пустые, мы полностью доверяем PAPI
    static inline void add_zgemm(int, bool = false, bool = false) noexcept {}
    static inline void add_zaxpy(int) noexcept {}
    static inline void add_zadd(int) noexcept {}
    static inline void add_zscal(int, bool = true) noexcept {}
    static inline void add_zlange(int) noexcept {}
};

// RAII кипер с использованием PAPI
class FlopScope
{
    uint64_t start_flop_cnt;
    bool prev_enabled;
    std::chrono::high_resolution_clock::time_point start_time;

    int event_set = PAPI_NULL;
    bool papi_active = false;

public:
    FlopScope() noexcept 
        : start_flop_cnt(FlopCounter::get_total_flops()),
          prev_enabled(FlopCounter::is_enabled()),
          start_time(std::chrono::high_resolution_clock::now())
    {
        FlopCounter::set_enabled(true);

        if (!PAPI_is_initialized()) {
            if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
                std::cerr << "PAPI library init error!\n";
                return;
            }
        }

        if (PAPI_create_eventset(&event_set) == PAPI_OK) {
            // Пытаемся добавить счетчик DP (двойная точность) FLOPS
            if (PAPI_add_event(event_set, PAPI_DP_OPS) == PAPI_OK) {
                if (PAPI_start(event_set) == PAPI_OK) {
                    papi_active = true;
                }
            } 
            // Fallback на общие FP_OPS
            else if (PAPI_add_event(event_set, PAPI_FP_OPS) == PAPI_OK) {
                if (PAPI_start(event_set) == PAPI_OK) {
                    papi_active = true;
                }
            }
            
            if (!papi_active) {
                std::cerr << "PAPI failed to start hardware counters!\n";
                PAPI_cleanup_eventset(event_set);
                PAPI_destroy_eventset(&event_set);
                event_set = PAPI_NULL;
            }
        }
    }

    ~FlopScope() noexcept
    {
        if (papi_active && event_set != PAPI_NULL) {
            long long counter_value = 0;
            // Фиксируем финальные значения и останавливаем счетчик
            if (PAPI_stop(event_set, &counter_value) == PAPI_OK) {
                FlopCounter::add_explicit(static_cast<uint64_t>(counter_value));
            }
            PAPI_cleanup_eventset(event_set);
            PAPI_destroy_eventset(&event_set);
        }
        FlopCounter::set_enabled(prev_enabled);
    }

    FlopScope(const FlopScope&) = delete;
    FlopScope& operator=(const FlopScope&) = delete;

    uint64_t flops() const noexcept
    {
        uint64_t current_hw_flops = 0;
        
        if (papi_active && event_set != PAPI_NULL) {
            long long counter_value = 0;
            // Читаем текущее значение счетчика "на лету" без его остановки
            if (PAPI_read(event_set, &counter_value) == PAPI_OK) {
                current_hw_flops = static_cast<uint64_t>(counter_value);
            }
        }
        
        // Суммируем программные флопсы (если остались в коде) и актуальные аппаратные
        return (FlopCounter::get_total_flops() - start_flop_cnt) + current_hw_flops;
    }

    double elapsed_ms() const noexcept
    {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }

    FlopStats stats() const noexcept
    {
        return { flops(), elapsed_ms() };
    }
};

} // namespace profiling