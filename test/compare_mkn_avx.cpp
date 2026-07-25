

#include "mkn/avx.hpp"
#include "mkn/kul/assert.hpp"

#include <vector>
#include <cstddef>

#include "bench.hpp"

// small enough that a+b stay resident in L1 across the whole run, so REPS
// exercises compute/instruction throughput rather than memory bandwidth
std::size_t constexpr static SIZE = 1024;
std::size_t constexpr static REPS = 4000000;

void test()
{
    auto constexpr static N = mkn::avx::Options::ALIGN();
    using Vec_t              = std::vector<double, mkn::kul::AlignedAllocator<double, N>>;

    Vec_t a(SIZE, 1), b(SIZE, 2);

    auto&& [sa, sb] = mkn::avx::make_spans(a, b);

    {
        Timer timer{SIZE * REPS};
        for (std::size_t r = 0; r < REPS; ++r)
            sb += sa;
    }

    mkn::kul::abort_if_not(b.front() == 2 + REPS);
    mkn::kul::abort_if_not(b.back() == 2 + REPS);
}

int main()
{
    std::cout << __FILE__ << std::endl;

    test();

    return 0;
}
