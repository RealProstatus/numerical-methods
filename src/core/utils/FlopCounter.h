#pragma once

#include <cstdint>
#include <chrono>
#include <iostream>

#if defined(HAS_PAPI)
#include <papi.h>
#endif

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

    // Заглушки для обратной совместимости, чтобы старый код компилировался без изменений
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

#if defined(HAS_PAPI)
    int event_set = PAPI_NULL;
    bool papi_active = false;
#endif

public:
    FlopScope() noexcept 
        : start_flop_cnt(FlopCounter::get_total_flops()),
          prev_enabled(FlopCounter::is_enabled()),
          start_time(std::chrono::high_resolution_clock::now())
    {
        FlopCounter::set_enabled(true);

#if defined(HAS_PAPI)
        if (PAPI_is_initialized() != PAPI_HIGH_LEVEL_INITED) {
            if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
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
            // Fallback на общие FP_OPS, если DP_OPS недоступен на уровне процессора в VM
            else if (PAPI_add_event(event_set, PAPI_FP_OPS) == PAPI_OK) {
                if (PAPI_start(event_set) == PAPI_OK) {
                    papi_active = true;
                }
            }
            
            if (!papi_active) {
                PAPI_destroy_eventset(&event_set);
                event_set = PAPI_NULL;
            }
        }
#endif
    }

    ~FlopScope() noexcept
    {
#if defined(HAS_PAPI)
        if (papi_active && event_set != PAPI_NULL) {
            long long counter_value = 0;
            if (PAPI_stop(event_set, &counter_value) == PAPI_OK) {
                FlopCounter::add_explicit(static_cast<uint64_t>(counter_value));
            }
            PAPI_destroy_eventset(&event_set);
        }
#endif
        FlopCounter::set_enabled(prev_enabled);
    }

    FlopScope(const FlopScope&) = delete;
    FlopScope& operator=(const FlopScope&) = delete;

    uint64_t flops() const noexcept
    {
        return FlopCounter::get_total_flops() - start_flop_cnt;
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
