

#include "xsimd/xsimd.hpp"
#include "mkn/kul/assert.hpp"

#include <vector>
#include <cstddef>

#include "bench.hpp"

namespace xs      = xsimd;
using vector_type = std::vector<double, xsimd::aligned_allocator<double>>;


// small enough that a+b stay resident in L1 across the whole run, so REPS
// exercises compute/instruction throughput rather than memory bandwidth
std::size_t constexpr static SIZE = 1024;
std::size_t constexpr static REPS = 4000000;

void test()
{
    vector_type a(SIZE, 1), b(SIZE, 2);

    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;

    {
        Timer timer{SIZE * REPS};
        for (std::size_t r = 0; r < REPS; ++r)
            for (std::size_t i = 0; i < SIZE; i += simd_size)
            {
                auto ba   = xs::load_aligned(&a[i]);
                auto bb   = xs::load_aligned(&b[i]);
                auto bres = ba + bb;
                bres.store_aligned(&b[i]);
            }
    }

    mkn::kul::abort_if_not(b.front() == 2 + REPS);
    mkn::kul::abort_if_not(b.back() == 2 + REPS);
}

int main()
{
    std::cout << __FILE__ << " " << xsimd::simd_type<double>::size << std::endl;

    test();

    return 0;
}
