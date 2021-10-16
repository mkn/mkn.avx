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
#ifndef _MKN_AVX_HPP_
#define _MKN_AVX_HPP_

#include <cstdint>
#include <utility>
#include <immintrin.h>  // avx

namespace mkn::avx {

template <typename T, std::size_t SIZE>
struct Type_ {
  Type_() = delete;  // generic class never to be instantiated.
};

template <typename Type>
auto& Type_set(Type& avx, typename Type::value_type val) noexcept {
  for (std::uint8_t i = 0; i < Type::value_count; i++) avx.array[i] = val;
  return avx.array;
}

template <typename T, std::size_t SIZE, typename Impl>
struct alignas(16) TypeDAO {
  static constexpr std::size_t value_count = SIZE;
  using value_type = T;
  using array_t = typename Impl::internal_type;

  TypeDAO() noexcept = default;
  TypeDAO(value_type val) noexcept : array{Type_set(*this, val)} {}
  TypeDAO(array_t&& arr) noexcept : array{arr} {}

  auto& operator[](std::size_t i) noexcept { return array[i]; }
  auto& operator[](std::size_t i) const noexcept { return array[i]; }
  auto& operator()() noexcept { return array; }
  auto& operator()() const noexcept { return array; }

  array_t array;
};

template <>
struct alignas(16) Type_<double, 2>{
  using internal_type = __m128d;
  auto constexpr static fma_func_ptr = &_mm_fmadd_pd;
};

template <>
struct alignas(16) Type_<double, 4>{
  using internal_type = __m256d;
  auto constexpr static fma_func_ptr = &_mm256_fmadd_pd;
};

template <>
struct alignas(16) Type_<float, 4>  {
  using internal_type = __m128;
  auto constexpr static fma_func_ptr = &_mm_fmadd_ps;
};

template <>
struct alignas(16) Type_<float, 8>   {
  using internal_type = __m256;
  auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
};

template <typename T, std::size_t SIZE>
struct Type : public TypeDAO<T, SIZE, Type_<T, SIZE>> {
  using DAO = TypeDAO<T, SIZE, Type_<T, SIZE>>;
  using array_t = typename DAO::array_t;
  auto constexpr static fma_func_ptr = Type_<T, SIZE>::fma_func_ptr;

  Type() noexcept = default;
  Type(typename DAO::value_type val) noexcept : DAO{val} {}
  Type(array_t&& arr) noexcept : DAO{std::forward<array_t>(arr)} {}
};

template <typename AVX_t>
AVX_t fma(AVX_t const& a, AVX_t const& b, AVX_t const& c){
  return {AVX_t::fma_func_ptr(a(), b(), c())};
}

} /* namespace mkn::avx */

namespace std {

template <typename T, std::size_t SIZE>
auto fma(mkn::avx::Type<T, SIZE> const& a, mkn::avx::Type<T, SIZE> const& b, mkn::avx::Type<T, SIZE> const& c) noexcept {
  return mkn::avx::fma(a, b, c);
}

} /* namespace std */

#endif /* _MKN_AVX_HPP_ */
