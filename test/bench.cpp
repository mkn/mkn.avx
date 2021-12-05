
#include <cassert>
#include "benchmark/benchmark.h"
#include "mkn/avx.hpp"

#include <cmath>

namespace mkn::noavx
{
template<uint8_t Max, typename Float>
void __attribute__((optimize("no-tree-vectorize")))
add(Float const* a, Float const* b, Float* c) noexcept
{
    // for (uint8_t i = 0; i < Max; i++)
    c[0] = a[0] + b[0];
}
template<uint8_t Max, typename Float>
void __attribute__((optimize("no-tree-vectorize")))
mul(Float const* a, Float const* b, Float* c) noexcept
{
    // for (uint8_t i = 0; i < Max; i++)
    c[0] = a[0] * b[0];
}
template<uint8_t Max, typename Float>
void __attribute__((optimize("no-tree-vectorize")))
multiply_and_add(Float const* a, Float const* b, Float const* c, Float* d) noexcept
{
    // for (uint8_t i = 0; i < Max; i++)
    d[0] = (a[0] * b[0]) + c[0];
}
} /* namespace mkn::noavx */

void mul_double(benchmark::State& state)
{
    std::size_t SIZE = 1000000;

    std::vector<double> a(SIZE, 2), b(SIZE, 2), c(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            mkn::noavx::mul<1>(&a[i], &b[i], &c[i]);
    for (std::size_t i = 0; i < SIZE; i++)
        assert(c[i] == 4);
}
BENCHMARK(mul_double)->Unit(benchmark::kMicrosecond);

void mul_double_vector(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<double> av(SIZE * AVX_COUNT, 2);
    std::vector<double> bv = av;
    std::vector<double> cv(SIZE * AVX_COUNT, 0);

    auto& a = *reinterpret_cast<std::vector<AVX>*>(&av);
    auto& b = *reinterpret_cast<std::vector<AVX>*>(&bv);
    auto& c = *reinterpret_cast<std::vector<AVX>*>(&cv);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            c[i] = a[i] * b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(c[0][i] == 4);
}
BENCHMARK(mul_double_vector)->Unit(benchmark::kMicrosecond);

void mul_double_avx_256(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 256 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 2), b(SIZE, 2), c(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            c[i] = a[i] * b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(c[0][i] == 4);
}
BENCHMARK(mul_double_avx_256)->Unit(benchmark::kMicrosecond);

void mul_double_avx_128(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 2), b(SIZE, 2), c(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            c[i] = a[i] * b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(c[0][i] == 4);
}
BENCHMARK(mul_double_avx_128)->Unit(benchmark::kMicrosecond);

void mul_double_avx_128_inplace(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1e6 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 1);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            a[i] *= b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(a[0][i] == 1);
}
BENCHMARK(mul_double_avx_128_inplace)->Unit(benchmark::kMicrosecond);

void add_double(benchmark::State& state)
{
    std::size_t SIZE = 1000000;

    std::vector<double> a(SIZE, 1), b(SIZE, 2), c(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            mkn::noavx::add<1>(&a[i], &b[i], &c[i]);
    for (std::size_t i = 0; i < SIZE; i++)
        assert(c[i] == 3);
}
BENCHMARK(add_double)->Unit(benchmark::kMicrosecond);

void add_double_avx_256(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 256 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            c[i] = a[i] + b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(c[0][i] == 3);
}
BENCHMARK(add_double_avx_256)->Unit(benchmark::kMicrosecond);

void add_double_avx_128(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            c[i] = a[i] + b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(c[0][i] == 3);
}
BENCHMARK(add_double_avx_128)->Unit(benchmark::kMicrosecond);

void add_double_avx_128_inplace(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1e6 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            a[i] += b[i];

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(int(a[0][i] + 1) % 2 == 0);
}
BENCHMARK(add_double_avx_128_inplace)->Unit(benchmark::kMicrosecond);

void fma_double(benchmark::State& state)
{
    std::size_t SIZE = 1000000;

    std::vector<double> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            mkn::noavx::multiply_and_add<1>(&a[i], &b[i], &c[i], &d[i]);
    for (std::size_t i = 0; i < SIZE; i++)
        assert(d[i] == 5);
}
BENCHMARK(fma_double)->Unit(benchmark::kMicrosecond);

void fma_double_avx_256(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 256 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            d[i] = std::fma(a[i], b[i], c[i]);

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(d[0][i] == 5);
}
BENCHMARK(fma_double_avx_256)->Unit(benchmark::kMicrosecond);

void fma_double_avx_128(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
    using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            d[i] = std::fma(a[i], b[i], c[i]);

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(d[0][i] == 5);
}
BENCHMARK(fma_double_avx_128)->Unit(benchmark::kMicrosecond);

void fma_float(benchmark::State& state)
{
    std::size_t SIZE = 1000000;

    std::vector<float> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            mkn::noavx::multiply_and_add<1>(&a[i], &b[i], &c[i], &d[i]);

    for (std::size_t i = 0; i < SIZE; i++)
        assert(d[i] == 5);
}
BENCHMARK(fma_float)->Unit(benchmark::kMicrosecond);

void fma_float_avx_256(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 256 / (sizeof(float) * 8);
    using AVX                       = mkn::avx::Type<float, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            d[i] = std::fma(a[i], b[i], c[i]);

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(d[0][i] == 5);
}
BENCHMARK(fma_float_avx_256)->Unit(benchmark::kMicrosecond);

void fma_float_avx_128(benchmark::State& state)
{
    constexpr std::size_t AVX_COUNT = 128 / (sizeof(float) * 8);
    using AVX                       = mkn::avx::Type<float, AVX_COUNT>;
    constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

    std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

    while (state.KeepRunning())
        for (std::size_t i = 0; i < SIZE; i++)
            d[i] = std::fma(a[i], b[i], c[i]);

    for (std::size_t i = 0; i < SIZE * AVX::value_count; ++i)
        assert(d[0][i] == 5);
}
BENCHMARK(fma_float_avx_128)->Unit(benchmark::kMicrosecond);
