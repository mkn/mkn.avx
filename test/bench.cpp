
#include <cassert>
#include "benchmark/benchmark.h"
#include "kul/avx.hpp"

#include <cmath>

namespace kul::noavx {
template <uint8_t Max, typename Float>
void __attribute__((optimize("no-tree-vectorize")))
multiply_and_add(Float const *a, Float const *b, Float const *c, Float *d) noexcept {
  // for (uint8_t i = 0; i < Max; i++)
  d[0] = (a[0] * b[0]) + c[0];
}
} /* namespace kul::noavx */

void fma_double(benchmark::State &state) {
  std::size_t SIZE = 1000000;

  std::vector<double> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

  while (state.KeepRunning())
    for (std::size_t i = 0; i < SIZE; i++)
      kul::noavx::multiply_and_add<1>(&a[i], &b[i], &c[i], &d[i]);
  for (std::size_t i = 0; i < SIZE; i++) assert(d[i] == 5);
}
BENCHMARK(fma_double)->Unit(benchmark::kMicrosecond);

void fma_double_avx_256(benchmark::State &state) {
  constexpr std::size_t AVX_COUNT = 256 / (sizeof(double) * 8);
  using AVX = kul::avx::Type<double, AVX_COUNT>;
  constexpr std::size_t SIZE = 1000000 / AVX::value_count;

  std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

  while (state.KeepRunning())
    for (std::size_t i = 0; i < SIZE; i++) d[i] = std::fma(a[i](), b[i](), c[i]());

  for (std::size_t i = 0; i < SIZE * AVX::value_count; i++) assert(d[0][i] == 5);
}
BENCHMARK(fma_double_avx_256)->Unit(benchmark::kMicrosecond);

void fma_double_avx_128(benchmark::State &state) {
  constexpr std::size_t AVX_COUNT = 128 / (sizeof(double) * 8);
  using AVX = kul::avx::Type<double, AVX_COUNT>;
  constexpr std::size_t SIZE = 1000000 / AVX::value_count;

  std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

  while (state.KeepRunning())
    for (std::size_t i = 0; i < SIZE; i++) d[i] = std::fma(a[i](), b[i](), c[i]());

  for (std::size_t i = 0; i < SIZE * AVX::value_count; i++) assert(d[0][i] == 5);
}
BENCHMARK(fma_double_avx_128)->Unit(benchmark::kMicrosecond);

void fma_float(benchmark::State &state) {
  std::size_t SIZE = 1000000;

  std::vector<float> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

  while (state.KeepRunning())
    for (std::size_t i = 0; i < SIZE; i++)
      kul::noavx::multiply_and_add<1>(&a[i], &b[i], &c[i], &d[i]);

  for (std::size_t i = 0; i < SIZE; i++) assert(d[i] == 5);
}
BENCHMARK(fma_float)->Unit(benchmark::kMicrosecond);

void fma_float_avx_256(benchmark::State &state) {
  constexpr std::size_t AVX_COUNT = 256 / (sizeof(float) * 8);
  using AVX = kul::avx::Type<float, AVX_COUNT>;
  constexpr std::size_t SIZE = 1000000 / AVX::value_count;

  std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

  while (state.KeepRunning())
    for (std::size_t i = 0; i < SIZE; i++) d[i] = std::fma(a[i](), b[i](), c[i]());

  for (std::size_t i = 0; i < SIZE * AVX::value_count; i++) assert(d[0][i] == 5);
}
BENCHMARK(fma_float_avx_256)->Unit(benchmark::kMicrosecond);

void fma_float_avx_128(benchmark::State &state) {
  constexpr std::size_t AVX_COUNT = 128 / (sizeof(float) * 8);
  using AVX = kul::avx::Type<float, AVX_COUNT>;
  constexpr std::size_t SIZE = 1000000 / AVX::value_count;

  std::vector<AVX> a(SIZE, 1), b(SIZE, 2), c(SIZE, 3), d(SIZE);

  while (state.KeepRunning())
    for (std::size_t i = 0; i < SIZE; i++) d[i] = std::fma(a[i](), b[i](), c[i]());

  for (std::size_t i = 0; i < SIZE * AVX::value_count; i++) assert(d[0][i] == 5);
}
BENCHMARK(fma_float_avx_128)->Unit(benchmark::kMicrosecond);
