
#include "mkn/kul/log.hpp"
#include "mkn/kul/math.hpp"
#include "mkn/avx.hpp"
#include "mkn/avx/vector.hpp"
#include "mkn/avx/grid.hpp"

#include <array>
#include <cassert>

template<typename T, std::uint16_t N>
void vec()
{
    mkn::avx::Vector<T> a(103, 2), b(103, 3);

    a += b;
    a *= b;

    assert(a == 15);
}

template<typename T, std::uint16_t N>
void span()
{
    mkn::avx::Vector<T> v0(103, 2);
    mkn::avx::Span<T, N> a{v0};

    mkn::avx::Vector<T> const v1(103, 3);
    mkn::avx::Span<T const, N> b{v1};

    a += b;
    assert(a == 5);
    a *= b;
    assert(a == 15);

    a -= 2;
    assert(a == 13);

    a -= b;
    assert(a == 10);

    // a /= b;
    // assert(a == 2);
}


template<typename T, std::uint16_t N>
void arr()
{
    using Span = mkn::avx::Span<T, N>;

    {
        std::array<T, N> b;
        std::vector<T> v0(N);
        auto a = mkn::avx::make_span(v0);

        for (std::size_t i = 0; i < N; ++i)
            v0[i] = (i + 1), b[i] = (i + 2);

        a += 2;
        a *= b;

        for (std::size_t i = 0; i < N; ++i)
            assert(v0[i] == ((i + 1) + 2) * (i + 2));
    }


    std::vector<T> v0(N, 2);
    Span a{v0};

    std::array<T, N> b;
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
        std::vector<T> v0(1000, 1), v1(1000, 1);

        mkn::avx::Grid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::Grid<T, 3> grid1{v1.data(), {10, 10, 10}};

        // (grid0 >> 0) += (grid1 >> 0);
        grid0 += grid1;
        KLOG(INF) << mkn::kul::math::sum(v0);
        assert(mkn::kul::math::sum(v0) == 2000);
    }

    // { // !!?? FAILS DUE TO ALIGNMENT ??!!
    //     static constexpr std::size_t S = 10;
    //     std::vector<T> v0(S*S*S, 1), v1(S*S*S, 1);

    //     mkn::avx::Grid<T, 3> grid0{v0.data(), {S, S, S}};
    //     mkn::avx::Grid<T, 3> grid1{v1.data(), {S, S, S}};

    //     (grid0 >> 0) += (grid1 >> 0);
    //     assert(mkn::kul::math::sum(v0) == S*S*S*2);
    // }

    {
        static constexpr std::size_t S = 12;
        std::vector<T> v0(S * S * S, 1), v1(S * S * S, 1);

        mkn::avx::Grid<T, 3> grid0{v0.data(), {S, S, S}};
        mkn::avx::Grid<T, 3> grid1{v1.data(), {S, S, S}};

        (grid0 >> 0) += (grid1 >> 0);
        // KLOG(INF) << mkn::kul::math::sum(v0);
        assert(mkn::kul::math::sum(v0) == S * S * S * 2);
    }

    // {
    //     std::vector<T> v0(1000, 1), v1(1000, 1);

    //     mkn::avx::Grid<T, 3> grid0{v0.data(), {10, 10, 10}};
    //     mkn::avx::Grid<T, 3> grid1{v1.data(), {10, 10, 10}};

    //     (grid0 >> 1) += (grid1 >> 1);
    //     assert(mkn::kul::math::sum(v0) == 1000 + (8 * 8 * 8));
    // }

    // {
    //     std::vector<T> v0(1000, 1), v1(1000, 2);

    //     mkn::avx::Grid<T, 3> grid0{v0.data(), {10, 10, 10}};
    //     mkn::avx::Grid<T, 3> grid1{v1.data(), {10, 10, 10}};

    //     (grid0 >> 1) *= (grid1 >> 1);
    //     assert(mkn::kul::math::sum(v0) == 1000 + (8 * 8 * 8));
    // }
}

template<typename T>
void test()
{
    // grid<T>();
    // arr<T>();
    // vec<T>();
}

int main() noexcept
{
    std::cout << __FILE__ << std::endl;

    span<double, 2>();
    span<double, 4>();

    span<float, 4>();
    span<float, 8>();

    return 0;
}
