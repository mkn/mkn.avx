
#if !defined(MKN_AVX_FN_COUNTER)
#define MKN_AVX_FN_COUNTER // optionally defined in dbg.hpp
#endif

#include <cstdint>
#include <utility>
#include <immintrin.h> // avx

namespace mkn::avx
{
template<typename T, std::size_t SIZE>
struct Type_
{
    static_assert(SIZE == 1); // otherwise missing impl

    using internal_type = T;

    // default operations without avx
    auto constexpr static add_func_ptr = [](auto& a, auto& b) { return a + b; };
    auto constexpr static sub_func_ptr = [](auto& a, auto& b) { return a - b; };
    auto constexpr static mul_func_ptr = [](auto& a, auto& b) { return a * b; };
    auto constexpr static div_func_ptr = [](auto& a, auto& b) { return a / b; };
    auto constexpr static set_func_ptr = [](auto a, auto& b) { return (*a) = b; };

    auto constexpr static set_v_func_ptr = [](auto& b) { return b; };

    auto constexpr static fma_func_ptr = [](auto& a, auto& b, auto& c) { return a * b + c; };
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
    using internal_type                  = __m128d;
    auto constexpr static add_func_ptr   = &_mm_add_pd;
    auto constexpr static sub_func_ptr   = &_mm_sub_pd;
    auto constexpr static mul_func_ptr   = &_mm_mul_pd;
    auto constexpr static div_func_ptr   = &_mm_div_pd;
    auto constexpr static set_func_ptr   = &_mm_store_pd;
    auto constexpr static set_v_func_ptr = &_mm_set1_pd;
    auto constexpr static fma_func_ptr   = &_mm_fmadd_pd;
};

template<>
struct Type_<double, 4>
{
    using internal_type                  = __m256d;
    auto constexpr static add_func_ptr   = &_mm256_add_pd;
    auto constexpr static sub_func_ptr   = &_mm256_sub_pd;
    auto constexpr static mul_func_ptr   = &_mm256_mul_pd;
    auto constexpr static div_func_ptr   = &_mm256_div_pd;
    auto constexpr static set_func_ptr   = &_mm256_store_pd;
    auto constexpr static set_v_func_ptr = &_mm256_set1_pd;

    auto constexpr static fma_func_ptr = &_mm256_fmadd_pd;
};

template<>
struct Type_<double, 8>
{
    using internal_type                  = __m512d;
    auto constexpr static add_func_ptr   = &_mm512_add_pd;
    auto constexpr static sub_func_ptr   = &_mm512_sub_pd;
    auto constexpr static mul_func_ptr   = &_mm512_mul_pd;
    auto constexpr static div_func_ptr   = &_mm512_div_pd;
    auto constexpr static set_func_ptr   = &_mm512_store_pd;
    auto constexpr static set_v_func_ptr = &_mm512_set1_pd;
    // auto constexpr static fma_func_ptr = &_mm256_fmadd_pd;
};
//////////////////// double ////////////////////




//////////////////// float ////////////////////
template<>
struct Type_<float, 4>
{
    using internal_type                  = __m128;
    auto constexpr static add_func_ptr   = &_mm_add_ps;
    auto constexpr static sub_func_ptr   = &_mm_sub_ps;
    auto constexpr static mul_func_ptr   = &_mm_mul_ps;
    auto constexpr static div_func_ptr   = &_mm_div_ps;
    auto constexpr static set_func_ptr   = &_mm_store_ps;
    auto constexpr static set_v_func_ptr = &_mm_set1_ps;
    auto constexpr static fma_func_ptr   = &_mm_fmadd_ps;
};

template<>
struct Type_<float, 8>
{
    using internal_type                  = __m256;
    auto constexpr static add_func_ptr   = &_mm256_add_ps;
    auto constexpr static sub_func_ptr   = &_mm256_sub_ps;
    auto constexpr static mul_func_ptr   = &_mm256_mul_ps;
    auto constexpr static div_func_ptr   = &_mm256_div_ps;
    auto constexpr static set_func_ptr   = &_mm256_store_ps;
    auto constexpr static set_v_func_ptr = &_mm256_set1_ps;
    auto constexpr static fma_func_ptr   = &_mm256_fmadd_ps;
};

template<>
struct Type_<float, 16>
{
    using internal_type                  = __m512;
    auto constexpr static add_func_ptr   = &_mm512_add_ps;
    auto constexpr static sub_func_ptr   = &_mm512_sub_ps;
    auto constexpr static mul_func_ptr   = &_mm512_mul_ps;
    auto constexpr static div_func_ptr   = &_mm512_div_ps;
    auto constexpr static set_func_ptr   = &_mm512_store_ps;
    auto constexpr static set_v_func_ptr = &_mm512_set1_ps;
    auto constexpr static fma_func_ptr   = &_mm512_fmadd_ps;
};

//////////////////// float ////////////////////




//////////////////// std::int16_t ////////////////////
template<>
struct Type_<std::int16_t, 4>
{
    using internal_type                = __m128i;
    auto constexpr static add_func_ptr = &_mm_add_epi16;
    auto constexpr static sub_func_ptr = &_mm_sub_epi16;
    // auto constexpr static mul_func_ptr = &_mm_mul_epi16;
    // auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
};
template<>
struct Type_<std::int16_t, 8>
{
    using internal_type                = __m256i;
    auto constexpr static add_func_ptr = &_mm256_add_epi16;
    auto constexpr static sub_func_ptr = &_mm256_sub_epi16;
    // auto constexpr static mul_func_ptr = &_mm256_mul_epi16;
    // auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
};
//////////////////// std::int16_t ////////////////////




//////////////////// std::int32_t ////////////////////
template<>
struct Type_<std::int32_t, 4>
{
    using internal_type                = __m128i;
    auto constexpr static add_func_ptr = &_mm_add_epi32;
    auto constexpr static mul_func_ptr = &_mm_mul_epi32;
    // auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
};
template<>
struct Type_<std::int32_t, 8>
{
    using internal_type                = __m256i;
    auto constexpr static add_func_ptr = &_mm256_add_epi32;
    auto constexpr static mul_func_ptr = &_mm256_mul_epi32;
    // auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
};
//////////////////// std::int32_t ////////////////////



// //////////////////// std::uint32_t ////////////////////
// template<>
// struct  Type_<std::uint32_t, 8>
// {
//     using internal_type                = __m256;
//     auto constexpr static add_func_ptr = &_mm256_add_epu32;
//     auto constexpr static mul_func_ptr = &_mm256_mul_epu32;
//     // auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
// };
// //////////////////// std::int32_t ////////////////////
// //////////////////// std::uint64_t ////////////////////
// template<>
// struct  Type_<std::int64_t, 4>
// {
//     using internal_type                = __m256;
//     auto constexpr static add_func_ptr = &_mm256_add_epi64;
//     auto constexpr static mul_func_ptr = &_mm256_mul_epi64;
//     // auto constexpr static fma_func_ptr = &_mm256_fmadd_ps;
// };
//////////////////// std::uint64_t ////////////////////



template<typename T, std::size_t SIZE>
using SuperType = TypeDAO<T, SIZE, Type_<T, SIZE>>;

template<typename T, std::size_t SIZE>
struct Type : public SuperType<T, SIZE>
{
    using Super      = SuperType<T, SIZE>;
    using value_type = typename Super::value_type;
    using array_t    = typename Super::array_t;

    auto constexpr static inline add_func_ptr   = Type_<T, SIZE>::add_func_ptr;
    auto constexpr static inline sub_func_ptr   = Type_<T, SIZE>::sub_func_ptr;
    auto constexpr static inline mul_func_ptr   = Type_<T, SIZE>::mul_func_ptr;
    auto constexpr static inline div_func_ptr   = Type_<T, SIZE>::div_func_ptr;
    auto constexpr static inline set_func_ptr   = Type_<T, SIZE>::set_func_ptr;
    auto constexpr static inline set_v_func_ptr = Type_<T, SIZE>::set_v_func_ptr;
    // auto constexpr static fma_func_ptr = Type_<T, SIZE>::fma_func_ptr;

    Type() noexcept = default;

    Type(array_t&& arr) noexcept
        : Super{std::forward<array_t>(arr)}
    {
    }
};

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator+(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::add_func_ptr(a(), b())};
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator-(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::Super::impl_type::sub_func_ptr(a(), b())};
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator*(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::mul_func_ptr(a(), b())};
}

template<typename T, std::size_t SIZE>
Type<T, SIZE> inline operator/(Type<T, SIZE> const& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    return {Type<T, SIZE>::Super::impl_type::div_func_ptr(a(), b())};
}

template<typename T, std::size_t SIZE>
void inline operator+=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::add_func_ptr(a(), b());
}

template<typename T, std::size_t SIZE>
void inline operator-=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::sub_func_ptr(a(), b());
}

template<typename T, std::size_t SIZE>
void inline operator*=(Type<T, SIZE>& __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::mul_func_ptr(a(), b());
}

template<typename T, std::size_t SIZE>
void inline operator/=(Type<T, SIZE>& a, Type<T, SIZE> const& b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::div_func_ptr(a(), b());
}


template<typename T, std::size_t SIZE>
void inline store(T const* __restrict a, Type<T, SIZE> const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    Type<T, SIZE>::set_func_ptr(const_cast<T*>(a), b());
}

template<typename T, std::size_t SIZE>
void inline store(Type<T, SIZE>& __restrict a, T const& __restrict b) noexcept
{
    MKN_AVX_FN_COUNTER;
    a() = Type<T, SIZE>::set_v_func_ptr(b);
}


template<typename T, std::size_t SIZE>
Type<T, SIZE> inline fma(Type<T, SIZE> const& a, Type<T, SIZE> const& b,
                         Type<T, SIZE> const& c) noexcept
{
    return {Type<T, SIZE>::Super::impl_type::fma_func_ptr(a(), b(), c())};
}

} /* namespace mkn::avx */
