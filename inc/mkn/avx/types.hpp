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

// #include "mkn/avx/dbg.hpp"

#include <cstdint>
#include <utility>
#include <immintrin.h> // avx


#if !defined(MKN_AVX_FN_COUNTER)
#define MKN_AVX_FN_COUNTER // optionally defined in dbg.hpp
#endif

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

    auto const static inline fma = _mm256_fmadd_pd;
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
    // auto const static inline fma = _mm256_fmadd_pd;
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
    using internal_type          = __m128i;
    auto const static inline add = [](auto&&... v) { return _mm_add_epi32(v...); };
    auto const static inline mul = [](auto&&... v) { return _mm_mul_epi32(v...); };
    // auto const static inline fma = _mm256_fmadd_ps;
};
template<>
struct Type_<std::int32_t, 8>
{
    using internal_type          = __m256i;
    auto const static inline add = [](auto&&... v) { return _mm256_add_epi32(v...); };
    auto const static inline mul = [](auto&&... v) { return _mm256_mul_epi32(v...); };
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
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::add(a(), b())};
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator-(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::Super::impl_type::sub(a(), b())};
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator*(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::mul(a(), b())};
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator/(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::Super::impl_type::div(a(), b())};
}

template<typename T, std::size_t SIZE>
void inline operator+=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::add(a(), b());
}

template<typename T, std::size_t SIZE>
void inline operator-=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::sub(a(), b());
}

template<typename T, std::size_t SIZE>
void inline operator*=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::mul(a(), b());
}

template<typename T, std::size_t SIZE>
void inline operator/=(Type<T, SIZE>& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::div(a(), b());
}


template<typename T, std::size_t SIZE>
void inline store(T* __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    Type<T, SIZE>::Super::impl_type::store(a, b());
}

template<typename T, std::size_t SIZE>
void inline store(Type<T, SIZE>& __restrict a, T const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::set_v(b);
}


template<typename T, std::size_t SIZE>
auto inline load(T const* __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    KEXCEPTION("load");
    // Type<T, SIZE>::load(const_cast<T*>(a), b());
}


template<typename T, std::size_t SIZE>
Type<T, SIZE> unaligned_load(T const* __restrict a) noexcept
{
    MKN_AVX_FN_COUNTER;
    return Type<T, SIZE>::Super::impl_type::unaligned_load(a /*const_cast<T*>(a)*/);
}

template<typename T, std::size_t SIZE>
void unaligned_store(T* __restrict a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    Type<T, SIZE>::Super::impl_type::unaligned_store(a, b());
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline fma(Type<T, SIZE> const& a, Type<T, SIZE> const& b,
                         Type<T, SIZE> const& c) noexcept
{
    return {Type<T, SIZE>::Super::impl_type::fma(a(), b(), c())};
}

} /* namespace mkn::avx */




#endif /* _MKN_AVX_TYPES_HPP_ */
