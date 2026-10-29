#pragma once

#include <cstdint>
#include <chrono>

namespace profiling
{

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

    static inline uint64_t get_total_flops() noexcept { return total_flop_cnt; }
    static inline bool is_enabled() noexcept { return enabled; }
    static inline void set_enabled(bool state) noexcept { enabled = state; }

    // Standart formulas for std::complex<double>

    static inline void add_zgemm(int N, bool has_alpha = false, bool has_betta = false)
    {
        if (!enabled)
        {
            return;
        }

        const uint64_t n = static_cast<uint64_t>(N);

        // N^3 complex mult (6 FLOP) + N^2*(N-1) complex add (2 FLOP)
        uint64_t ops = 8 * n*n*n - 2 * n*n;
        if (has_alpha)
        {
            ops += 6 * n*n;
        }
        if (has_betta)
        {
            ops += 8 * n*n;
        }

        total_flop_cnt += ops;
    }

    static inline void add_zaxpy(int N) noexcept
    {
        add(8 * static_cast<uint64_t>(N));
    }

    static inline void add_zadd(int N) noexcept
    {
        add(2 * static_cast<uint64_t>(N));
    }

    // X = alpha * X (умножение комплексного числа на комплексное = 6 FLOP, на вещественное = 2 FLOP)
    static inline void add_zscal(int N, bool complex_scalar = true) noexcept
    {
        add((complex_scalar ? 6 : 2) * static_cast<uint64_t>(N));
    }

    // 1-норма матрицы: N^2 |z| = sqrt(re^2 + im^2) (4 FLOP) + N(N-1) сумма по столбцам (1 FLOP)
    static inline void add_zlange(int N) noexcept
    {
        const uint64_t n2 = static_cast<uint64_t>(N) * static_cast<uint64_t>(N);
        add(5 * n2 - static_cast<uint64_t>(N));
    }
};

// RAII keeper of FlopCounter
class FlopScope
{
    uint64_t start_flop_cnt;
    bool prev_enabled;
    std::chrono::high_resolution_clock::time_point start_time;

public:
    FlopScope() noexcept :  start_flop_cnt(FlopCounter::get_total_flops()),
                            prev_enabled(FlopCounter::is_enabled()),
                            start_time(std::chrono::high_resolution_clock::now())
    {
        FlopCounter::set_enabled(true);
    }

    ~FlopScope() noexcept
    {
        FlopCounter::set_enabled(prev_enabled);
    }

    // Copy is prohibited
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