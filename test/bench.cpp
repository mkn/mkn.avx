
#include <cmath>
#include <array>
#include <cassert>

#include "mkn/kul/log.hpp"

#include "mkn/avx.hpp"
#include "mkn/avx/vector.hpp"

#include "benchmark/benchmark.h"

std::size_t constexpr SIZE = 1e6;

#if !defined(__INTEL_COMPILER) && !defined(__clang__) && defined(__GNUC__)
#if (__GNUC__ == 4 && __GNUC_MINOR__ > 3) || (__GNUC__ >= 5)
#define NO_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#else
#define NO_VECTORIZE _Pragma("GCC optimize(\"no-tree-vectorize\")")
#endif
#else
#define NO_VECTORIZE
#endif

namespace mkn::noavx
{
template<std::uint64_t SIZE, typename Float>
void inline NO_VECTORIZE add(Float const* a, Float const* b, Float* c) noexcept
{
#pragma clang loop vectorize(disable)
    for (std::size_t i = 0; i < SIZE; ++i)
        c[i] = a[i] + b[i];
}
template<std::uint64_t SIZE, typename Float>
void inline NO_VECTORIZE add_inplace(Float* a, Float const* b) noexcept
{
#pragma clang loop vectorize(disable)
    for (std::size_t i = 0; i < SIZE; ++i)
        a[i] += b[i];
}


template<std::uint64_t SIZE, typename Float>
void inline NO_VECTORIZE mul(Float const* a, Float const* b, Float* c) noexcept
{
#pragma clang loop vectorize(disable)
    for (std::size_t i = 0; i < SIZE; ++i)
        c[i] = a[i] * b[i];
}
template<std::uint64_t SIZE, typename Float>
void inline NO_VECTORIZE mul_inplace(Float* a, Float const* b) noexcept
{
#pragma clang loop vectorize(disable)
    for (std::size_t i = 0; i < SIZE; ++i)
        a[i] *= b[i];
}

template<std::uint64_t SIZE, typename Float>
void inline NO_VECTORIZE multiply_and_add(Float const* a, Float const* b, Float const* c,
                                          Float* d) noexcept
{
    d[0] = (a[0] * b[0]) + c[0];
}
} /* namespace mkn::noavx */


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void mul_fp64_noavx(benchmark::State& state)
{
    std::vector<double> a(SIZE, 2), b(SIZE, 2), c(SIZE);

    for (auto _ : state)
        mkn::noavx::mul<SIZE>(&a[0], &b[0], &c[0]);
}
BENCHMARK(mul_fp64_noavx)->Unit(benchmark::kMicrosecond);

void mul_fp64_noavx_inplace(benchmark::State& state)
{
    std::vector<double> a(SIZE, 2), b(SIZE, 2);

    for (auto _ : state)
        mkn::noavx::mul_inplace<SIZE>(&a[0], &b[0]);
}
BENCHMARK(mul_fp64_noavx_inplace)->Unit(benchmark::kMicrosecond);


void mul_fp64_avx(benchmark::State& state)
{
    mkn::avx::Vector<double> a(SIZE, 2), b(SIZE, 2), c(SIZE);
    for (auto _ : state)
        c.mul(a, b);
}
BENCHMARK(mul_fp64_avx)->Unit(benchmark::kMicrosecond);


void mul_fp64_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<double> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a *= b;
}
BENCHMARK(mul_fp64_avx_inplace)->Unit(benchmark::kMicrosecond);



void mul_fp32_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<float> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a *= b;
}
BENCHMARK(mul_fp32_avx_inplace)->Unit(benchmark::kMicrosecond);


void mul_std_int32_t_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<std::int32_t> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a *= b;
}
BENCHMARK(mul_std_int32_t_avx_inplace)->Unit(benchmark::kMicrosecond);




//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////



void add_fp64_noavx(benchmark::State& state)
{
    std::vector<double> a(SIZE, 1), b(SIZE, 2), c(SIZE);

    for (auto _ : state)
        mkn::noavx::add<SIZE>(&a[0], &b[0], &c[0]);
}
BENCHMARK(add_fp64_noavx)->Unit(benchmark::kMicrosecond);


void add_fp64_noavx_inplace(benchmark::State& state)
{
    std::vector<double> a(SIZE, 1), b(SIZE, 2);

    for (auto _ : state)
        mkn::noavx::add_inplace<SIZE>(&a[0], &b[0]);
}
BENCHMARK(add_fp64_noavx_inplace)->Unit(benchmark::kMicrosecond);


void add_fp64_avx(benchmark::State& state)
{
    mkn::avx::Vector<double> a(SIZE, 2), b(SIZE, 2), c(SIZE);
    for (auto _ : state)
        c.add(a, b);
}
BENCHMARK(add_fp64_avx)->Unit(benchmark::kMicrosecond);

void add_fp64_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<double> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a += b;
}
BENCHMARK(add_fp64_avx_inplace)->Unit(benchmark::kMicrosecond);


void add_fp32_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<float> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a *= b;
}
BENCHMARK(add_fp32_avx_inplace)->Unit(benchmark::kMicrosecond);


void add_std_int32_t_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<std::int32_t> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a += b;
}
BENCHMARK(add_std_int32_t_avx_inplace)->Unit(benchmark::kMicrosecond);




//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


// void fma_double(benchmark::State& state)
// {

//     std::vector<double> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

//     for (auto _ : state)
//         for (std::size_t i = 0; i < SIZE; i++)
//             mkn::noavx::multiply_and_add<1>(&a[i], &b[i], &c[i], &d[i]);
// }
// BENCHMARK(fma_double)->Unit(benchmark::kMicrosecond);

// void fma_fp64_avx_256(benchmark::State& state)
// {
//     constexpr std::size_t AVX_COUNT = 256 / (sizeof(double) * 8);
//     using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
//     constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

//     std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

//     for (auto _ : state)
//         for (std::size_t i = 0; i < SIZE; i++)
//             d[i] = std::fma(a[i], b[i], c[i]);
// }
// BENCHMARK(fma_fp64_avx_256)->Unit(benchmark::kMicrosecond);

// void fma_fp64_avx_128(benchmark::State& state)
// {
//     constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
//     using AVX                       = mkn::avx::Type<double, AVX_COUNT>;
//     constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

//     std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

//     for (auto _ : state)
//         for (std::size_t i = 0; i < SIZE; i++)
//             d[i] = std::fma(a[i], b[i], c[i]);
// }
// BENCHMARK(fma_fp64_avx_128)->Unit(benchmark::kMicrosecond);

// void fma_float(benchmark::State& state)
// {
//     std::size_t SIZE = 1000000;

//     std::vector<float> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

//     for (auto _ : state)
//         for (std::size_t i = 0; i < SIZE; i++)
//             mkn::noavx::multiply_and_add<1>(&a[i], &b[i], &c[i], &d[i]);
// }
// BENCHMARK(fma_float)->Unit(benchmark::kMicrosecond);

// void fma_fp32_avx_256(benchmark::State& state)
// {
//     constexpr std::size_t AVX_COUNT = 256 / (sizeof(float) * 8);
//     using AVX                       = mkn::avx::Type<float, AVX_COUNT>;
//     constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

//     std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

//     for (auto _ : state)
//         for (std::size_t i = 0; i < SIZE; i++)
//             d[i] = std::fma(a[i], b[i], c[i]);
// }
// BENCHMARK(fma_fp32_avx_256)->Unit(benchmark::kMicrosecond);

// void fma_fp32_avx_128(benchmark::State& state)
// {
//     constexpr std::size_t AVX_COUNT = 128 / (sizeof(float) * 8);
//     using AVX                       = mkn::avx::Type<float, AVX_COUNT>;
//     constexpr std::size_t SIZE      = 1000000 / AVX::value_count;

//     std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

//     for (auto _ : state)
//         for (std::size_t i = 0; i < SIZE; i++)
//             d[i] = std::fma(a[i], b[i], c[i]);
// }
// BENCHMARK(fma_fp32_avx_128)->Unit(benchmark::kMicrosecond);



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
