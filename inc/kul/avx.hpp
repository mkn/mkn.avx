/**
Copyright (c) 2020, Philip Deegan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Philip Deegan nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _KUL_AVX_HPP_
#define _KUL_AVX_HPP_

#include <cstdint>
#include <immintrin.h>  // avx

namespace kul::avx {

template <typename T, std::size_t SIZE>
struct Type {
  Type() = delete;  // generic class never to be instantiated.
};

template <typename Type>
auto& Type_set(Type& avx, typename Type::value_type val) noexcept {
  for (std::uint8_t i = 0; i < Type::value_count; i++) avx.array[i] = val;
  return avx.array;
}

template <>
struct alignas(16) Type<double, 2> {
  static constexpr std::size_t value_count = 2;
  using value_type = double;
  using internal_type = __m128d;
  using fma_array = internal_type[3];
  Type() noexcept = default;
  Type(value_type val) noexcept : array{Type_set(*this, val)} {}
  Type(internal_type&& arr) noexcept : array{arr} {}
  auto& operator[](std::size_t i) noexcept { return array[i]; }
  auto& operator[](std::size_t i) const noexcept { return array[i]; }
  auto& operator()() noexcept { return array; }
  auto& operator()() const noexcept { return array; }
  internal_type array;
};

template <>
struct alignas(16) Type<double, 4> {
  static constexpr std::size_t value_count = 4;
  using value_type = double;
  using internal_type = __m256d;
  using fma_array = internal_type[3];
  Type() noexcept = default;
  Type(value_type val) noexcept : array{Type_set(*this, val)} {}
  Type(internal_type&& arr) noexcept : array{arr} {}
  auto& operator[](std::size_t i) noexcept { return array[i]; }
  auto& operator[](std::size_t i) const noexcept { return array[i]; }
  auto& operator()() noexcept { return array; }
  auto& operator()() const noexcept { return array; }
  internal_type array;
};

template <>
struct alignas(16) Type<float, 4> {
  static constexpr std::size_t value_count = 4;
  using value_type = float;
  using internal_type = __m128;
  using fma_array = internal_type[3];
  Type() = default;
  Type(value_type val) noexcept : array{Type_set(*this, val)} {}
  Type(internal_type&& arr) noexcept : array{arr} {}
  auto& operator[](std::size_t i) noexcept { return array[i]; }
  auto& operator[](std::size_t i) const noexcept { return array[i]; }
  auto& operator()() noexcept { return array; }
  auto& operator()() const noexcept { return array; }
  internal_type array;
};

template <>
struct alignas(16) Type<float, 8> {
  static constexpr std::size_t value_count = 8;
  using value_type = float;
  using internal_type = __m256;
  using fma_array = internal_type[3];
  Type() noexcept = default;
  Type(value_type val) noexcept : array{Type_set(*this, val)} {}
  Type(internal_type&& arr) noexcept : array{arr} {}
  auto& operator[](std::size_t i) noexcept { return array[i]; }
  auto& operator[](std::size_t i) const noexcept { return array[i]; }
  auto& operator()() noexcept { return array; }
  auto& operator()() const noexcept { return array; }
  internal_type array;
};

} /* namespace kul::avx */

namespace std {
inline auto fma(__m128 const& a, __m128 const& b, __m128 const& c) noexcept {
  return _mm_fmadd_ps(a, b, c);
}
inline auto fma(__m256 const& a, __m256 const& b, __m256 const& c) noexcept {
  return _mm256_fmadd_ps(a, b, c);
}

inline auto fma(__m256d const& a, __m256d const& b, __m256d const& c) noexcept {
  return _mm256_fmadd_pd(a, b, c);
}
inline auto fma(__m128d const& a, __m128d const& b, __m128d const& c) {
  return _mm_fmadd_pd(a, b, c);
}

} /* namespace std */

#endif /* _KUL_AVX_HPP_ */
