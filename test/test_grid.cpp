
#include "mkn/kul/log.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/kul/math.hpp"
#include "mkn/kul/float.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/avx.hpp"
#include "mkn/avx/grid.hpp"

#include <cassert>


// shape is exactly N per dimension, so the whole flattened grid - and every
// full row - is a clean multiple of N: satisfies Grid's exact-division
// contract, no NestedGrid slicing (which would reintroduce a ragged tail).
template<typename T = double>
void grid_clean()
{
    static constexpr std::size_t N = mkn::avx::Options::N<T>();

    mkn::avx::Vector<T> v0(N * N * N, 1), v1(N * N * N, 1);

    mkn::avx::Grid<T, 3> grid0{v0.data(), {N, N, N}};
    mkn::avx::Grid<T, 3> grid1{v1.data(), {N, N, N}};

    grid0 += grid1;

    mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), (N * N * N) * 2));
}

// arbitrary shapes / offset slices - row remainders aren't guaranteed to
// divide evenly by N, so these need AsymmetricGrid's leftover handling.
template<typename T = double>
void grid_asymmetric()
{
    {
        mkn::avx::Vector<T> v0(1000, 1), v1(1000, 1);

        mkn::avx::AsymmetricGrid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::AsymmetricGrid<T, 3> grid1{v1.data(), {10, 10, 10}};

        grid0 += grid1;

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), 2000));
    }

    {
        static constexpr std::size_t S = 12;
        mkn::avx::Vector<T> v0(S * S * S, 1), v1(S * S * S, 1);

        mkn::avx::AsymmetricGrid<T, 3> grid0{v0.data(), {S, S, S}};
        mkn::avx::AsymmetricGrid<T, 3> grid1{v1.data(), {S, S, S}};

        (grid0 >> 0) += (grid1 >> 0);

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), S * S * S * 2));
    }

    {
        mkn::avx::Vector<T> v0(1000, 0);
        mkn::avx::Vector<T> const v1(1000, 1);

        mkn::avx::AsymmetricGrid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::AsymmetricGrid<T const, 3> const grid1{v1.data(), {10, 10, 10}};

        (grid0 >> 1) += (grid1 >> 1);

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), 8 * 8 * 8));
    }

    {
        std::vector<T> v0(1000, 0), v1(1000, 2);

        mkn::avx::AsymmetricGrid<T, 3> grid0{v0.data(), {10, 10, 10}};
        mkn::avx::AsymmetricGrid<T, 3> grid1{v1.data(), {10, 10, 10}};

        (grid0 >> 1) += (grid1 >> 1);
        (grid0 >> 1) *= (grid1 >> 1);

        mkn::kul::abort_if_not(mkn::kul::float_equals(mkn::kul::math::sum(v0), 4 * (8 * 8 * 8)));
    }
}

template<typename T>
void test()
{
    grid_clean<T>();
    grid_asymmetric<T>();
}

int main() noexcept
{
    KOUT(NON) << __FILE__;

    test<float>();
    test<double>();

    return 0;
}
