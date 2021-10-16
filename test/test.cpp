
#include "mkn/avx.hpp"

#include <array>
#include <vector>
#include <cassert>
#include <iostream>

int main() noexcept {
  std::cout << __FILE__ << std::endl;

  using AVX = mkn::avx::Type<double, 4>;
  static constexpr std::size_t SIZE = AVX::value_count * 3;
  static_assert((SIZE % AVX::value_count) == 0, "Bad AVX value_count for vector");

  std::vector<AVX> a(SIZE / AVX::value_count, 1);
  for (size_t i = 0; i < SIZE; i++) assert(a[0][i] == 1);  // dangeresque

  a[1][0] = 2;
  a[2][0] = 3;

  a[0][3] = 2;
  a[1][3] = 2;
  a[2][3] = 3;

  auto check = [](auto&& v) {
    assert(v[0] == (1 * 2) + 3);
    assert(v[1] == (1 * 1) + 1);
    assert(v[2] == (1 * 1) + 1);
    assert(v[3] == (2 * 2) + 3);
  };

  check(std::fma(a[0], a[1], a[2]));

  return 0;
}
