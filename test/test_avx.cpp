
#include "mkn/avx.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/avx/vector.hpp"

#include <cmath>
#include <cassert>
#include <iostream>


template<typename T>
void array()
{
    using Array_t = mkn::avx::Array<T, mkn::avx::Options::N<T>()>;
    Array_t a{1}, b{2};

    assert(a == 1);
    assert(b == 2);

    a += b;
    assert(a == 3);

    a *= b;
    assert(a == 6);

    auto const c = []() {
        Array_t t1{2};
        Array_t t2{1};
        return t1 + t2;
    }();

    a *= c;
    assert(a == 18);

    b = a;
    assert(b == 18);
}




template<typename T>
void span()
{
    using Vec = mkn::avx::Vector_t<T>;
    Vec v0(103, 2);
    auto a = mkn::avx::make_span(v0);

    Vec const v1(103, 3);
    auto b = mkn::avx::make_span(v1);

    a += b;
    a *= b;
    assert(a == 15);

    // a /= b;
    // a -= b;
    // assert(a == 2);
}


template<typename T>
void arr()
{
    using Array_t    = mkn::avx::Array<T, mkn::avx::Options::N<T>()>;
    using Vec        = mkn::avx::Vector_t<T>;
    constexpr auto N = mkn::avx::Span<T>::N;
    {
        Array_t b;
        Vec v0(N);
        auto a = mkn::avx::make_span(v0);

        for (std::size_t i = 0; i < N; ++i)
            v0[i] = (i + 1), b[i] = (i + 2);

        a += 2;
        a *= b;

        for (std::size_t i = 0; i < N; ++i)
            assert(v0[i] == ((i + 1) + 2) * (i + 2));
    }

    Vec v0(N * 10, 2);
    auto a = mkn::avx::make_span(v0);

    Array_t b;
    std::fill(b.begin(), b.end(), 3);

    a *= b;
    assert(a == 6);
}


template<typename T = double>
void fma()
{
    using AVX                         = mkn::avx::Type<T, 4>;
    static constexpr std::size_t SIZE = AVX::value_count * 3;
    static_assert((SIZE % AVX::value_count) == 0, "Bad AVX value_count for vector");

    std::vector<AVX> a(SIZE / AVX::value_count, 1);
    for (size_t i = 0; i < SIZE; i++)
    {
        assert(a[0][i] == 1); // dangeresque
    }

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
}

template<typename T>
void test()
{
    array<T>();
    span<T>();
    arr<T>();
}

int main() noexcept
{
    std::cout << __FILE__ << std::endl;
    test<float>();
    test<double>();

    for (auto const& [k, v] : mkn::avx::Counter::I().cnts)
    {
        KLOG(INF) << k << " " << v;
    }
    return 0;
}
