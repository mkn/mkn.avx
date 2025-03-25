


#include "mkn/avx.hpp"
#include "mkn/avx/def.hpp"
#include "mkn/avx/span.hpp"
#include "mkn/kul/dbg.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/avx/vector.hpp"

#include <cstdint>
#include <tuple>
#include <type_traits>


template<typename T>
struct Vec
{
    Vec(std::size_t const s)
        : vec(s)
    {
    }

    auto& operator()() { return vec; }

    auto size() const { return vec.size(); }
    auto data() { return vec.data(); }
    auto data() const { return vec.data(); }

    mkn::avx::Vector_t<T> vec;
};

template<typename T, typename V = T>
struct SoA
{
    std::size_t s = 10;
    Vec<V> v0{s}, v1{s}, v2{s}, v3{s};

    operator bool() { return v0().size() == s; }
    auto size() const { return s; }

    void operator()();
};

template<typename T>
struct AoS
{
    std::size_t s = 10;

    operator bool() { return v().size() == s; }

    void operator()();

    struct E
    {
        T v0, v1, v2, v3;
    };


    auto size() const { return v.size(); }
    auto data() { return v.data(); }
    auto data() const { return v.data(); }

    Vec<E> v{s};
};

template<typename T = double, typename... Args>
void op(Args&&... args)
{
    auto&& [a, b, c, d] = std::forward_as_tuple(args...);

    T t0 = 2;
    a    = 10;
    c    = 400;
    T t1 = 2;
    T t3 = t0 * t1;
    b += t3;
    c += t1;

    T t4 = 100;
    t4 *= t3;
    d = a + b * c;
    d += t4;
    d *= a;
    d *= b;
    d *= c;
}

template<typename T>
void AoS<T>::operator()()
{
    KUL_DBG_FUNC_ENTER;
    if constexpr (std::is_same_v<double, T>)
    {
        for (std::size_t i = 0; i < size(); ++i)
            op<T>(v.data()[i].v0, v.data()[i].v1, v.data()[i].v2, v.data()[i].v3);
    }
    else
    {
        auto constexpr static N = mkn::avx::Options::N<T>();

        // NO AVX FOR AOS
        for (std::size_t i = 0; i < size(); i += N)
        {
            for (std::size_t j = 0; j < N; ++j)
            {
                std::size_t off = i + j;
                op<T>(v.data()[off].v0, v.data()[off].v1, v.data()[off].v2, v.data()[off].v3);
            }
        }
    }
}

template<typename T, typename V>
void SoA<T, V>::operator()()
{
    KUL_DBG_FUNC_ENTER;
    if constexpr (std::is_same_v<double, T>)
    {
        for (std::size_t i = 0; i < size(); ++i)
            op<T>(v0().data()[i], v1().data()[i], v2().data()[i], v3().data()[i]);
    }
    else
    {
        auto constexpr static N = mkn::avx::Options::N<V>();

        // AVX!!
        for (std::size_t i = 0; i < size(); i += N)
        {
            auto&& [a0, a1, a2, a3] = mkn::avx::make_spans<N>(i, v0(), v1(), v2(), v3());
            op<T>(a0, a1, a2, a3);
        }
    }
}