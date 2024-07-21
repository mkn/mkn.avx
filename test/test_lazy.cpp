
#include "mkn/avx/lazy.hpp"
#include "mkn/kul/assert.hpp"

using namespace mkn::avx;

void fma_simple()
{
    using DV      = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    std::size_t N = 4;
    DV a0(N, 1), a1(N, 2), a2(N, 3);
    auto [l0, l1, l2] = lazy(a0, a1, a2);
    auto lz           = l0 * l1 + l2;
    auto r            = eval(lz);
    KOUT(NON) << r.front() << " " << r.back();
    mkn::kul::abort_if_not(r.front() == 5 and r.back() == 5);
}

void fma0()
{
    using DV      = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    std::size_t N = 4;
    DV a0(N, 1), a1(N, 2), a2(N, 3);
    auto [l0, l1, l2] = lazy(a0, a1, a2);
    auto lz           = l0 * l1 + l2 + l0;
    auto r            = eval(lz);
    KOUT(NON) << r.front() << " " << r.back();
    mkn::kul::abort_if_not(r.front() == 6 and r.back() == 6);
}

void fma1()
{
    using DV      = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    std::size_t N = 4;
    DV a0(N, 1), a1(N, 2), a2(N, 3);
    auto [l0, l1, l2] = lazy(a0, a1, a2);
    auto lz           = l0 * l1 + l2 + l0 * l1 + l2;
    auto r            = eval(lz);
    KOUT(NON) << r.front() << " " << r.back();
    mkn::kul::abort_if_not(r.front() == 10 and r.back() == 10);
}

void fma()
{
    fma_simple();
    fma0();
    fma1();
}


void fn0()
{
    using DV      = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    std::size_t N = 1e6;
    DV a0(N, 1), a1(N, 2), a2(N, 3), a3(N, 4), a4(N, 5);
    assert(a0.data() and a4.data());
    auto [l0, l1, l2, l3, l4] = lazy(a0, a1, a2, a3, a4);
    auto lz                   = l0 * l1 + l2 * l3 + l4 * l1 + l2 * l3 + l4 * l1;
    auto r                    = eval(lz);
    KOUT(NON) << r.front() << " " << r.back();
    mkn::kul::abort_if_not(r.front() == 46 and r.back() == 46);
}

void fn1()
{
    using DV      = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    std::size_t N = 1e6;
    DV a0(N, 1), a1(N, 2), a2(N, 3), a3(N, 4), a4(N, 5);
    assert(a0.data() and a4.data());
    auto [l0, l1, l2, l3, l4] = lazy(a0, a1, a2, a3, a4);
    auto r                    = eval(l0 * l1 + l2 + l3 + l4 * l1 + l2 * l3 + l4 * l1);
    KOUT(NON) << r.front() << " " << r.back();
    mkn::kul::abort_if_not(r.front() == 41 and r.back() == 41);
}

int main()
{
    // fma();
    fn0();
    fn1();
};
