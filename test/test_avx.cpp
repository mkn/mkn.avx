
#include "mkn/kul/log.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/kul/math.hpp"
#include "mkn/kul/float.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/avx.hpp"
#include "mkn/avx/grid.hpp"

#include <cassert>



template<typename T>
void array()
{
    using Array_t = mkn::avx::Array<T, mkn::avx::Options::N<T>()>;
    Array_t a{1}, b{2};

    mkn::kul::abort_if_not(a == 1);
    mkn::kul::abort_if_not(b == 2);

    a += b;
    mkn::kul::abort_if_not(a == 3);

    a *= b;
    mkn::kul::abort_if_not(a == 6);

    auto const c = []() {
        Array_t t1{2};
        Array_t t2{1};
        return t1 + t2;
    }();

    a *= c;
    mkn::kul::abort_if_not(a == 18);
}




template<typename T, std::uint16_t N = mkn::avx::Options::N<T>()>
void span()
{
    using Vec = typename mkn::avx::Vector_t<T>;

    Vec v0(103, 2);
    auto a = mkn::avx::make_span(v0);

    Vec const v1(103, 3);
    auto b = mkn::avx::make_span(v1);

    a += b;
    assert(a == 5);
    a *= b;
    assert(a == 15);

    a -= 2;
    assert(a == 13);

    a -= b;
    assert(a == 10);

    a += 2;

    a /= b;
    assert(a == 4);
}


template<typename T, std::uint16_t N = mkn::avx::Options::N<T>()>
void arr()
{
    using Array_t = mkn::avx::Array<T, mkn::avx::Options::N<T>()>;
    using Vec     = mkn::avx::Vector_t<T>;

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

    a += b;
    assert(a == 5);
    a *= b;
    assert(a == 15);
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


template<typename T = double>
void grid()
{
    {
        mkn::avx::Vector_t<T> v0(1000, 1), v1(1000, 1);

        mkn::avx::Grid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::Grid<T, 3> grid1{v1.data(), {10, 10, 10}};

        grid0 += grid1;

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), 2000));
    }

    {
        static constexpr std::size_t S = 12;
        mkn::avx::Vector_t<T> v0(S * S * S, 1), v1(S * S * S, 1);

        mkn::avx::Grid<T, 3> grid0{v0.data(), {S, S, S}};
        mkn::avx::Grid<T, 3> grid1{v1.data(), {S, S, S}};

        (grid0 >> 0) += (grid1 >> 0);

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), S * S * S * 2));
    }

    {
        mkn::avx::Vector_t<T> v0(1000, 0);
        mkn::avx::Vector_t<T> const v1(1000, 1);

        mkn::avx::Grid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::Grid<T const, 3> const grid1{v1.data(), {10, 10, 10}};

        (grid0 >> 1) += (grid1 >> 1);

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), 8 * 8 * 8));
    }

    {
        std::vector<T> v0(1000, 0), v1(1000, 2);

        mkn::avx::Grid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::Grid<T, 3> grid1{v1.data(), {10, 10, 10}};

        (grid0 >> 1) += (grid1 >> 1);
        (grid0 >> 1) *= (grid1 >> 1);

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), 4 * (8 * 8 * 8)));
    }
}

template<typename T>
void test()
{
    array<T>();
    arr<T>();
    grid<T>();
    span<T>();
}

int main() noexcept
{
    KOUT(NON) << __FILE__;

    test<float>();
    test<double>();


    return 0;
}
