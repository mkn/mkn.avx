
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


template<typename T>
void mul_no_avx(benchmark::State& state)
{
    std::vector<double> a(SIZE, 2), b(SIZE, 2), c(SIZE);

    for (auto _ : state)
        mkn::noavx::mul<SIZE>(&a[0], &b[0], &c[0]);
}


template<typename T>
void mul_no_avx_inplace(benchmark::State& state)
{
    std::vector<double> a(SIZE, 2), b(SIZE, 2);

    for (auto _ : state)
        mkn::noavx::mul_inplace<SIZE>(&a[0], &b[0]);
}



template<typename T>
void mul_avx(benchmark::State& state)
{
    mkn::avx::Vector<T> a(SIZE, 2), b(SIZE, 2), c(SIZE);
    for (auto _ : state)
        c.mul(a, b);
}


template<typename T>
void mul_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<T> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a *= b;
}


template<typename T>
void mul_avx_inplace_single(benchmark::State& state)
{
    mkn::avx::Vector<T> a(SIZE, 2);
    for (auto _ : state)
        a *= 2;
}


template<typename T>
void mul_avx_inplace_array(benchmark::State& state)
{
    constexpr auto N = mkn::avx::Span<T>::N;

    mkn::avx::Vector<T> a(SIZE, 2);
    std::array<T, N> b;
    std::fill(b.begin(), b.end(), 2);

    for (auto _ : state)
        a *= b;
}


template<typename T>
void add_avx_inplace(benchmark::State& state)
{
    mkn::avx::Vector<T> a(SIZE, 2), b(SIZE, 2);
    for (auto _ : state)
        a += b;
}


template<typename T>
void add_avx_inplace_single(benchmark::State& state)
{
    mkn::avx::Vector<T> a(SIZE, 2);
    for (auto _ : state)
        a += 2;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


BENCHMARK_TEMPLATE(mul_no_avx, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_no_avx_inplace, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace_array, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace_single, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(add_avx_inplace, double)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(add_avx_inplace_single, double)->Unit(benchmark::kMicrosecond);


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


BENCHMARK_TEMPLATE(mul_no_avx, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_no_avx_inplace, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace_array, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace_single, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(add_avx_inplace, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(add_avx_inplace_single, float)->Unit(benchmark::kMicrosecond);



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


BENCHMARK_TEMPLATE(mul_avx_inplace, std::uint32_t)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace_array, std::uint32_t)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mul_avx_inplace_single, std::uint32_t)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(add_avx_inplace, std::uint32_t)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(add_avx_inplace_single, std::uint32_t)->Unit(benchmark::kMicrosecond);


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
