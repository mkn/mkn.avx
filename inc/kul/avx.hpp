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

template <typename T, std::size_t SIZE>
struct alignas(16) TypeDAO {
  static constexpr std::size_t value_count = SIZE;
  using value_type = T;
};

template <>
struct alignas(16) Type<double, 2> : TypeDAO<double, 2> {
  using internal_type = __m128d;
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
struct alignas(16) Type<double, 4> : TypeDAO<double, 4> {
  using internal_type = __m256d;
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
struct alignas(16) Type<float, 4> : TypeDAO<float, 4> {
  using internal_type = __m128;
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
struct alignas(16) Type<float, 8> : TypeDAO<float, 8> {
  using internal_type = __m256;
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
inline auto fma(kul::avx::Type<float, 4> const& a, kul::avx::Type<float, 4> const& b, kul::avx::Type<float, 4> const& c) noexcept {
  return kul::avx::Type<float, 4>{_mm_fmadd_ps(a(), b(), c())};
}
inline auto fma(kul::avx::Type<float, 8> const& a, kul::avx::Type<float, 8> const& b, kul::avx::Type<float, 8> const& c) noexcept {
  return kul::avx::Type<float, 8>{_mm256_fmadd_ps(a(), b(), c())};
}

inline auto fma(kul::avx::Type<double, 2> const& a, kul::avx::Type<double, 2> const& b, kul::avx::Type<double, 2> const& c) noexcept {
  return kul::avx::Type<double, 2>{_mm_fmadd_pd(a(), b(), c())};
}
inline auto fma(kul::avx::Type<double, 4> const& a, kul::avx::Type<double, 4> const& b, kul::avx::Type<double, 4> const& c) noexcept {
  return kul::avx::Type<double, 4>{_mm256_fmadd_pd(a(), b(), c())};
}

} /* namespace std */

#endif /* _KUL_AVX_HPP_ */
