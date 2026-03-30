#include <vector>
#include <cstddef>
#include "xsimd/xsimd.hpp"
#include "mkn/kul/assert.hpp"

#include "bench.hpp"

namespace xs      = xsimd;
using vector_type = std::vector<double, xsimd::aligned_allocator<double>>;


std::size_t constexpr static COUNT = 10000000;

void test()
{
    vector_type a(COUNT, 1), b(COUNT, 2);

    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;

    {
        Timer timer{COUNT};
        for (std::size_t i = 0; i < COUNT; i += simd_size)
        {
            auto ba   = xs::load_aligned(&a[i]);
            auto bb   = xs::load_aligned(&b[i]);
            auto bres = ba + bb;
            bres.store_aligned(&b[i]);
        }
    }

    mkn::kul::abort_if_not(b.front() == 3);
    mkn::kul::abort_if_not(b.back() == 3);
}

int main()
{
    std::cout << __FILE__ << " " << xsimd::simd_type<double>::size << std::endl;

    test();

    return 0;
}
