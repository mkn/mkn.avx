/**
Copyright (c) 2024, Philip Deegan.
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
#ifndef _MKN_AVX_TYPES_HPP_
#define _MKN_AVX_TYPES_HPP_

// IWYU pragma: private, include "mkn/avx.hpp"
// IWYU pragma: friend ".*mkn/avx/.*"

#include <immintrin.h> // avx
#include <cstdint>
#include <utility>
#include <cmath>
#include <cstddef>

namespace mkn::avx
{
template<typename T, std::size_t SIZE>
struct Type_
{
    static_assert(SIZE == 1); // otherwise missing impl

    using internal_type = T;

    // default operations without avx
    auto constexpr static add                = [](auto& a, auto& b) { return a + b; };
    auto constexpr static sub                = [](auto& a, auto& b) { return a - b; };
    auto constexpr static mul                = [](auto& a, auto& b) { return a * b; };
    auto constexpr static div                = [](auto& a, auto& b) { return a / b; };
    auto constexpr static store              = [](auto a, auto& b) { return (*a) = b; };
    auto constexpr static set_v              = [](auto& b) { return b; };
    auto const static inline unaligned_load  = [](auto a) { return *a; };
    auto const static inline unaligned_store = [](auto a, auto& b) { return *a = b; };

    auto constexpr static fma = [](auto& a, auto& b, auto& c) { return a * b + c; };

    // masks are bools
    using mask_type                = bool;
    auto constexpr static floor    = [](auto const& a) { return std::floor(a); };
    auto constexpr static abs      = [](auto const& a) { return std::abs(a); };
    auto constexpr static lt       = [](auto const& a, auto const& b) { return a < b; };
    auto constexpr static gt       = [](auto const& a, auto const& b) { return a > b; };
    auto constexpr static mask_and = [](bool const a, bool const b) { return a and b; };
    auto constexpr static mask_or  = [](bool const a, bool const b) { return a or b; };
    auto constexpr static any      = [](bool const m) { return m; };
    auto constexpr static select
        = [](bool const m, auto const& a, auto const& b) { return m ? a : b; };
    auto constexpr static to_int32   = [](auto const& a) { return static_cast<std::int32_t>(a); };
    auto constexpr static from_int32 = [](std::int32_t const a) { return static_cast<T>(a); };
    // masked: a false lane neither reads nor writes memory
    auto constexpr static mask_load  = [](bool const m, auto const* p) { return m ? *p : T{}; };
    auto constexpr static mask_store = [](bool const m, auto* p, auto const& v) {
        if (m)
            *p = v;
    };
};



template<typename T, std::size_t SIZE, typename Impl>
struct TypeDAO
{
    static constexpr std::size_t value_count = SIZE;
    using value_type                         = T;
    using impl_type                          = Impl;
    using array_t                            = typename Impl::internal_type;

    TypeDAO() noexcept = default;

    TypeDAO(array_t&& arr) noexcept
        : array{arr}
    {
    }

    auto& operator[](std::size_t i) noexcept { return reinterpret_cast<T*>(&array)[i]; }
    auto& operator[](std::size_t i) const noexcept { return array[i]; }
    inline auto& operator()() noexcept { return array; }
    inline auto& operator()() const noexcept { return array; }
    inline auto data() { return array.data(); }
    inline auto data() const { return array.data(); }

    array_t array;
};


//////////////////// double ////////////////////
template<>
struct Type_<double, 2>
{
    using internal_type                      = __m128d;
    auto const static inline add             = [](auto&&... v) { return _mm_add_pd(v...); };
    auto const static inline sub             = [](auto&&... v) { return _mm_sub_pd(v...); };
    auto const static inline mul             = [](auto&&... v) { return _mm_mul_pd(v...); };
    auto const static inline div             = [](auto&&... v) { return _mm_div_pd(v...); };
    auto const static inline store           = [](auto&&... v) { return _mm_store_pd(v...); };
    auto const static inline set_v           = [](auto&&... v) { return _mm_set1_pd(v...); };
    auto const static inline fma             = [](auto&&... v) { return _mm_fmadd_pd(v...); };
    auto const static inline unaligned_load  = [](auto&&... v) { return _mm_loadu_pd(v...); };
    auto const static inline unaligned_store = [](auto&&... v) { return _mm_storeu_pd(v...); };

    using mask_type                = __m128d; // all bits set per true lane
    auto const static inline floor = [](auto const& a) { return _mm_floor_pd(a); };
    auto const static inline abs
        = [](auto const& a) { return _mm_andnot_pd(_mm_set1_pd(-0.0), a); };
    auto const static inline lt = [](auto const& a, auto const& b) { return _mm_cmplt_pd(a, b); };
    auto const static inline gt = [](auto const& a, auto const& b) { return _mm_cmpgt_pd(a, b); };
    auto const static inline mask_and
        = [](auto const& a, auto const& b) { return _mm_and_pd(a, b); };
    auto const static inline mask_or = [](auto const& a, auto const& b) { return _mm_or_pd(a, b); };
    auto const static inline any     = [](auto const& m) { return _mm_movemask_pd(m) != 0; };
    auto const static inline select
        = [](auto const& m, auto const& a, auto const& b) { return _mm_blendv_pd(b, a, m); };
    // masked (AVX): a false lane neither reads nor writes memory, loads 0
    auto const static inline mask_load
        = [](auto const& m, double const* p) { return _mm_maskload_pd(p, _mm_castpd_si128(m)); };
    auto const static inline mask_store = [](auto const& m, double* p, auto const& v) {
        _mm_maskstore_pd(p, _mm_castpd_si128(m), v);
    };
};

template<>
struct Type_<double, 4>
{
    using internal_type                      = __m256d;
    auto const static inline add             = [](auto&&... v) { return _mm256_add_pd(v...); };
    auto const static inline sub             = [](auto&&... v) { return _mm256_sub_pd(v...); };
    auto const static inline mul             = [](auto&&... v) { return _mm256_mul_pd(v...); };
    auto const static inline div             = [](auto&&... v) { return _mm256_div_pd(v...); };
    auto const static inline store           = [](auto&&... v) { return _mm256_store_pd(v...); };
    auto const static inline set_v           = [](auto&&... v) { return _mm256_set1_pd(v...); };
    auto const static inline unaligned_load  = [](auto&&... v) { return _mm256_loadu_pd(v...); };
    auto const static inline unaligned_store = [](auto&&... v) { return _mm256_storeu_pd(v...); };

    using mask_type                = __m256d; // all bits set per true lane
    auto const static inline floor = [](auto const& a) { return _mm256_floor_pd(a); };
    auto const static inline abs
        = [](auto const& a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a); };
    auto const static inline lt
        = [](auto const& a, auto const& b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); };
    auto const static inline gt
        = [](auto const& a, auto const& b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); };
    auto const static inline mask_and
        = [](auto const& a, auto const& b) { return _mm256_and_pd(a, b); };
    auto const static inline mask_or
        = [](auto const& a, auto const& b) { return _mm256_or_pd(a, b); };
    auto const static inline any = [](auto const& m) { return _mm256_movemask_pd(m) != 0; };
    auto const static inline select
        = [](auto const& m, auto const& a, auto const& b) { return _mm256_blendv_pd(b, a, m); };
    // masked: a false lane neither reads nor writes memory, loads 0
    auto const static inline mask_load = [](auto const& m, double const* p) {
        return _mm256_maskload_pd(p, _mm256_castpd_si256(m));
    };
    auto const static inline mask_store = [](auto const& m, double* p, auto const& v) {
        _mm256_maskstore_pd(p, _mm256_castpd_si256(m), v);
    };
    // truncating, to and from 4 int32 lanes (Type_<std::int32_t, 4>)
    auto const static inline to_int32   = [](auto const& a) { return _mm256_cvttpd_epi32(a); };
    auto const static inline from_int32 = [](auto const& a) { return _mm256_cvtepi32_pd(a); };

    // a lambda, not the intrinsic's address: MSVC's intrinsics have none (LNK2019)
    auto const static inline fma = [](auto&&... v) { return _mm256_fmadd_pd(v...); };
};

template<>
struct Type_<double, 8>
{
    using internal_type                      = __m512d;
    auto const static inline add             = [](auto&&... v) { return _mm512_add_pd(v...); };
    auto const static inline sub             = [](auto&&... v) { return _mm512_sub_pd(v...); };
    auto const static inline mul             = [](auto&&... v) { return _mm512_mul_pd(v...); };
    auto const static inline div             = [](auto&&... v) { return _mm512_div_pd(v...); };
    auto const static inline store           = [](auto&&... v) { return _mm512_store_pd(v...); };
    auto const static inline set_v           = [](auto&&... v) { return _mm512_set1_pd(v...); };
    auto const static inline unaligned_load  = [](auto&&... v) { return _mm512_loadu_pd(v...); };
    auto const static inline unaligned_store = [](auto&&... v) { return _mm512_storeu_pd(v...); };
    auto const static inline fma             = [](auto&&... v) { return _mm512_fmadd_pd(v...); };

    using mask_type                = __mmask8; // a bit per lane
    auto const static inline floor = [](auto const& a) {
        return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    };
    auto const static inline abs = [](auto const& a) { return _mm512_abs_pd(a); };
    auto const static inline lt
        = [](auto const& a, auto const& b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); };
    auto const static inline gt
        = [](auto const& a, auto const& b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); };
    auto const static inline mask_and
        = [](__mmask8 const a, __mmask8 const b) { return static_cast<__mmask8>(a & b); };
    auto const static inline mask_or
        = [](__mmask8 const a, __mmask8 const b) { return static_cast<__mmask8>(a | b); };
    auto const static inline any    = [](__mmask8 const m) { return m != 0; };
    auto const static inline select = [](__mmask8 const m, auto const& a, auto const& b) {
        return _mm512_mask_blend_pd(m, b, a);
    };
    // masked: a false lane neither reads nor writes memory, loads 0
    auto const static inline mask_load
        = [](__mmask8 const m, double const* p) { return _mm512_maskz_loadu_pd(m, p); };
    auto const static inline mask_store
        = [](__mmask8 const m, double* p, auto const& v) { _mm512_mask_storeu_pd(p, m, v); };
    // truncating, to and from 8 int32 lanes (Type_<std::int32_t, 8>)
    auto const static inline to_int32   = [](auto const& a) { return _mm512_cvttpd_epi32(a); };
    auto const static inline from_int32 = [](auto const& a) { return _mm512_cvtepi32_pd(a); };
};
//////////////////// double ////////////////////




//////////////////// float ////////////////////
template<>
struct Type_<float, 4>
{
    using internal_type                      = __m128;
    auto const static inline add             = [](auto&&... v) { return _mm_add_ps(v...); };
    auto const static inline sub             = [](auto&&... v) { return _mm_sub_ps(v...); };
    auto const static inline mul             = [](auto&&... v) { return _mm_mul_ps(v...); };
    auto const static inline div             = [](auto&&... v) { return _mm_div_ps(v...); };
    auto const static inline store           = [](auto&&... v) { return _mm_store_ps(v...); };
    auto const static inline set_v           = [](auto&&... v) { return _mm_set1_ps(v...); };
    auto const static inline fma             = [](auto&&... v) { return _mm_fmadd_ps(v...); };
    auto const static inline unaligned_load  = [](auto&&... v) { return _mm_loadu_ps(v...); };
    auto const static inline unaligned_store = [](auto&&... v) { return _mm_storeu_ps(v...); };
};

template<>
struct Type_<float, 8>
{
    using internal_type                      = __m256;
    auto const static inline add             = [](auto&&... v) { return _mm256_add_ps(v...); };
    auto const static inline sub             = [](auto&&... v) { return _mm256_sub_ps(v...); };
    auto const static inline mul             = [](auto&&... v) { return _mm256_mul_ps(v...); };
    auto const static inline div             = [](auto&&... v) { return _mm256_div_ps(v...); };
    auto const static inline store           = [](auto&&... v) { return _mm256_store_ps(v...); };
    auto const static inline set_v           = [](auto&&... v) { return _mm256_set1_ps(v...); };
    auto const static inline fma             = [](auto&&... v) { return _mm256_fmadd_ps(v...); };
    auto const static inline unaligned_load  = [](auto&&... v) { return _mm256_loadu_ps(v...); };
    auto const static inline unaligned_store = [](auto&&... v) { return _mm256_storeu_ps(v...); };
};

template<>
struct Type_<float, 16>
{
    using internal_type                      = __m512;
    auto const static inline add             = [](auto&&... v) { return _mm512_add_ps(v...); };
    auto const static inline sub             = [](auto&&... v) { return _mm512_sub_ps(v...); };
    auto const static inline mul             = [](auto&&... v) { return _mm512_mul_ps(v...); };
    auto const static inline div             = [](auto&&... v) { return _mm512_div_ps(v...); };
    auto const static inline store           = [](auto&&... v) { return _mm512_store_ps(v...); };
    auto const static inline set_v           = [](auto&&... v) { return _mm512_set1_ps(v...); };
    auto const static inline fma             = [](auto&&... v) { return _mm512_fmadd_ps(v...); };
    auto const static inline unaligned_load  = [](auto&&... v) { return _mm512_loadu_ps(v...); };
    auto const static inline unaligned_store = [](auto&&... v) { return _mm512_storeu_ps(v...); };
};

//////////////////// float ////////////////////




//////////////////// std::int16_t ////////////////////
template<>
struct Type_<std::int16_t, 4>
{
    using internal_type          = __m128i;
    auto const static inline add = [](auto&&... v) { return _mm_add_epi16(v...); };
    auto const static inline sub = [](auto&&... v) { return _mm_sub_epi16(v...); };
    // auto const static inline mul = _mm_mul_epi16;
    // auto const static inline fma = _mm256_fmadd_ps;
};
template<>
struct Type_<std::int16_t, 8>
{
    using internal_type          = __m256i;
    auto const static inline add = [](auto&&... v) { return _mm256_add_epi16(v...); };
    auto const static inline sub = [](auto&&... v) { return _mm256_sub_epi16(v...); };
    // auto const static inline mul = _mm256_mul_epi16;
    // auto const static inline fma = _mm256_fmadd_ps;
};
//////////////////// std::int16_t ////////////////////




//////////////////// std::int32_t ////////////////////
template<>
struct Type_<std::int32_t, 4>
{
    using internal_type            = __m128i;
    auto const static inline add   = [](auto&&... v) { return _mm_add_epi32(v...); };
    auto const static inline mul   = [](auto&&... v) { return _mm_mullo_epi32(v...); };
    auto const static inline set_v = [](std::int32_t const v) { return _mm_set1_epi32(v); };
    auto const static inline sub   = [](auto&&... v) { return _mm_sub_epi32(v...); };
    auto const static inline store
        = [](std::int32_t* p, auto const& v) { _mm_store_si128(reinterpret_cast<__m128i*>(p), v); };
    // no integer division instruction: an error only where used
    auto const static inline div = [](auto&&... v) {
        static_assert(sizeof...(v) == 0, "mkn::avx: no int32 lane division");
        return __m128i{};
    };
    auto const static inline unaligned_load = [](std::int32_t const* p) {
        return _mm_loadu_si128(reinterpret_cast<__m128i const*>(p));
    };
    auto const static inline unaligned_store = [](std::int32_t* p, auto const& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    };
    // auto const static inline fma = _mm256_fmadd_ps;
};
template<>
struct Type_<std::int32_t, 8>
{
    using internal_type            = __m256i;
    auto const static inline add   = [](auto&&... v) { return _mm256_add_epi32(v...); };
    auto const static inline mul   = [](auto&&... v) { return _mm256_mullo_epi32(v...); };
    auto const static inline set_v = [](std::int32_t const v) { return _mm256_set1_epi32(v); };
    auto const static inline sub   = [](auto&&... v) { return _mm256_sub_epi32(v...); };
    auto const static inline store = [](std::int32_t* p, auto const& v) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    };
    // no integer division instruction: an error only where used
    auto const static inline div = [](auto&&... v) {
        static_assert(sizeof...(v) == 0, "mkn::avx: no int32 lane division");
        return __m256i{};
    };
    auto const static inline unaligned_load = [](std::int32_t const* p) {
        return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(p));
    };
    auto const static inline unaligned_store = [](std::int32_t* p, auto const& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    };
    // auto const static inline fma = _mm256_fmadd_ps;
};
//////////////////// std::int32_t ////////////////////



// //////////////////// std::uint32_t ////////////////////
// template<>
// struct  Type_<std::uint32_t, 8>
// {
//     using internal_type                = __m256;
//     auto const static inline add = _mm256_add_epu32;
//     auto const static inline mul = _mm256_mul_epu32;
//     // auto const static inline fma = _mm256_fmadd_ps;
// };
// //////////////////// std::int32_t ////////////////////
// //////////////////// std::uint64_t ////////////////////
// template<>
// struct  Type_<std::int64_t, 4>
// {
//     using internal_type                = __m256;
//     auto const static inline add = _mm256_add_epi64;
//     auto const static inline mul = _mm256_mul_epi64;
//     // auto const static inline fma = _mm256_fmadd_ps;
// };
//////////////////// std::uint64_t ////////////////////



template<typename T, std::size_t SIZE>
using SuperType = TypeDAO<T, SIZE, Type_<T, SIZE>>;

template<typename T, std::size_t SIZE>
struct Type : public SuperType<T, SIZE>
{
    using Super      = SuperType<T, SIZE>;
    using value_type = Super::value_type;
    using array_t    = Super::array_t;

    auto const static inline add   = Type_<T, SIZE>::add;
    auto const static inline sub   = Type_<T, SIZE>::sub;
    auto const static inline mul   = Type_<T, SIZE>::mul;
    auto const static inline div   = Type_<T, SIZE>::div;
    auto const static inline store = Type_<T, SIZE>::store;
    auto const static inline set_v = Type_<T, SIZE>::set_v;
    // auto constexpr static fma = Type_<T, SIZE>::fma;

    Type() noexcept = default;

    Type(array_t&& arr) noexcept
        : Super{std::forward<array_t>(arr)}
    {
    }
};


template<typename T, std::size_t SIZE>
struct UnType : public Type<T, SIZE> // Unaligned flag
{
};

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator+(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{ return {Type<T, SIZE>::add(a(), b())}; }

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator-(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{ return {Type<T, SIZE>::Super::impl_type::sub(a(), b())}; }

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator*(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{ return {Type<T, SIZE>::mul(a(), b())}; }

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator/(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{ return {Type<T, SIZE>::Super::impl_type::div(a(), b())}; }

template<typename T, std::size_t SIZE>
void inline operator+=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{ a() = Type<T, SIZE>::add(a(), b()); }

template<typename T, std::size_t SIZE>
void inline operator-=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{ a() = Type<T, SIZE>::sub(a(), b()); }

template<typename T, std::size_t SIZE>
void inline operator*=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{ a() = Type<T, SIZE>::mul(a(), b()); }

template<typename T, std::size_t SIZE>
void inline operator/=(Type<T, SIZE>& a, Type<T, SIZE> const& b) noexcept
{ a() = Type<T, SIZE>::div(a(), b()); }

template<typename T, std::size_t SIZE>
auto inline load(T const a) noexcept
{ return Type<T, SIZE>::set_v(a); }

template<typename T, std::size_t SIZE>
void inline store(T* __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{ Type<T, SIZE>::Super::impl_type::store(a, b()); }

template<typename T, std::size_t SIZE>
Type<T, SIZE> unaligned_load(T const* __restrict a) noexcept
{ return Type<T, SIZE>::Super::impl_type::unaligned_load(a); }

template<typename T, std::size_t SIZE>
void unaligned_store(T* __restrict a, Type<T, SIZE> const& b) noexcept
{ Type<T, SIZE>::Super::impl_type::unaligned_store(a, b()); }

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline fma(Type<T, SIZE> const& a, Type<T, SIZE> const& b,
                         Type<T, SIZE> const& c) noexcept
{ return {Type<T, SIZE>::Super::impl_type::fma(a(), b(), c())}; }

// a lane mask, from the comparisons: per lane true or false (SSE / AVX: a vector with all bits of
//  a true lane set, AVX-512: a bit per lane, scalar: a bool)
template<typename T, std::size_t SIZE>
struct Mask
{
    using internal_type = typename Type_<T, SIZE>::mask_type;
    internal_type m;
};

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline floor(Type<T, SIZE> const& a) noexcept
{ return {Type_<T, SIZE>::floor(a())}; }

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline abs(Type<T, SIZE> const& a) noexcept
{ return {Type_<T, SIZE>::abs(a())}; }

// a < b and a > b per lane (ordered: false for NaN)
template<typename T, std::size_t SIZE>
Mask<T, SIZE> inline lt(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{ return {Type_<T, SIZE>::lt(a(), b())}; }

template<typename T, std::size_t SIZE>
Mask<T, SIZE> inline gt(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{ return {Type_<T, SIZE>::gt(a(), b())}; }

template<typename T, std::size_t SIZE>
Mask<T, SIZE> inline operator&(Mask<T, SIZE> const& a, Mask<T, SIZE> const& b) noexcept
{ return {Type_<T, SIZE>::mask_and(a.m, b.m)}; }

template<typename T, std::size_t SIZE>
Mask<T, SIZE> inline operator|(Mask<T, SIZE> const& a, Mask<T, SIZE> const& b) noexcept
{ return {Type_<T, SIZE>::mask_or(a.m, b.m)}; }

// true if any lane is
template<typename T, std::size_t SIZE>
bool inline any(Mask<T, SIZE> const& m) noexcept
{ return Type_<T, SIZE>::any(m.m); }

// per lane: a where m, else b
template<typename T, std::size_t SIZE>
Type<T, SIZE> inline select(Mask<T, SIZE> const& m, Type<T, SIZE> const& a,
                            Type<T, SIZE> const& b) noexcept
{ return {Type_<T, SIZE>::select(m.m, a(), b())}; }

// unaligned, masked: the lanes of m read (0 for the others) or written, the others' memory not
//  touched (may be past the end of an array)
template<typename T, std::size_t SIZE>
Type<T, SIZE> inline mask_load(Mask<T, SIZE> const& m, T const* p) noexcept
{ return {Type_<T, SIZE>::mask_load(m.m, p)}; }

template<typename T, std::size_t SIZE>
void inline mask_store(Mask<T, SIZE> const& m, T* p, Type<T, SIZE> const& v) noexcept
{ Type_<T, SIZE>::mask_store(m.m, p, v()); }

// truncated to std::int32_t lanes, and back
template<std::size_t SIZE>
Type<std::int32_t, SIZE> inline to_int32(Type<double, SIZE> const& a) noexcept
{ return {Type_<double, SIZE>::to_int32(a())}; }

template<std::size_t SIZE>
Type<double, SIZE> inline to_double(Type<std::int32_t, SIZE> const& a) noexcept
{ return {Type_<double, SIZE>::from_int32(a())}; }

} /* namespace mkn::avx */


#endif /* _MKN_AVX_TYPES_HPP_ */
