
#include "mkn/avx/lazy.hpp"
#include "mkn/kul/assert.hpp"

using namespace mkn::avx;

constexpr static std::size_t N = 1e6 + 5;

void add()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2);
    auto [l0, l1] = lazy(a0, a1);
    auto r        = eval(l0 + l1 + l0);

    mkn::kul::abort_if_not(r.front() == 4 and r.back() == 4);
}


void mul()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2);
    auto [l0, l1] = lazy(a0, a1);
    auto r        = eval(l0 * l1 * l0);

    mkn::kul::abort_if_not(r.front() == 2 and r.back() == 2);
}

void fma3()
{
    {
        using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
        DV a0(N, 1), a1(N, 2), a2(N, 3);
        auto [l0, l1, l2] = lazy(a0, a1, a2);
        auto lz           = l0 * l1 + l2;
        auto r            = eval(lz);

        mkn::kul::abort_if_not(r.front() == 5 and r.back() == 5);
    }
    {
        using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
        DV a0(N, 1), a1(N, 2), a2(N, 3);
        auto [l0, l1, l2] = lazy(a0, a1, a2);
        auto lz           = l0 + l1 * l2;
        auto r            = eval(lz);

        mkn::kul::abort_if_not(r.front() == 7 and r.back() == 7);
    }
}

void fma0()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2), a2(N, 3);
    auto [l0, l1, l2] = lazy(a0, a1, a2);
    auto lz           = l0 * l1 + l2 + l0;
    auto r            = eval(lz);

    mkn::kul::abort_if_not(r.front() == 6 and r.back() == 6);
}

void fma1()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2), a2(N, 3), a3(N, 4), a4(N, 5);
    auto [l0, l1, l2, l3, l4] = lazy(a0, a1, a2, a3, a4);
    auto r                    = eval(l0 * l1 + l2 * l3 + l4 * l1 + l2 * l3 + l4 * l1);

    mkn::kul::abort_if_not(r.front() == 46 and r.back() == 46);
}
void fma2()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2), a2(N, 3), a3(N, 4), a4(N, 5), a5(N, 6);
    auto [l0, l1, l2, l3, l4, l5] = lazy(a0, a1, a2, a3, a4, a5);
    auto r                        = eval(l0 * l1 + l2 + l3 + l4 * l5);

    mkn::kul::abort_if_not(r.front() == 39 and r.back() == 39);
}

void fma() {}


void fn0()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2), a2(N, 3), a3(N, 4), a4(N, 5);
    assert(a0.data() and a4.data());
    auto [l0, l1, l2, l3, l4] = lazy(a0, a1, a2, a3, a4);
    auto lz                   = l0 * l1 + l2 * l3 + l4 * l1 + l2 * l3 + l4 * l1;
    auto r                    = eval(lz);

    mkn::kul::abort_if_not(r.front() == 46 and r.back() == 46);
}

void fn1()
{
    using DV = std::vector<double, mkn::kul::AlignedAllocator<double, 32>>;
    DV a0(N, 1), a1(N, 2), a2(N, 3), a3(N, 4), a4(N, 5);
    assert(a0.data() and a4.data());
    auto [l0, l1, l2, l3, l4] = lazy(a0, a1, a2, a3, a4);
    auto r                    = eval(l0 * l1 + l2 + l3 + l4 * l1 + l2 * l3 + l4 * l1);

    mkn::kul::abort_if_not(r.front() == 41 and r.back() == 41);
}

int main()
{
    add();
    mul();
    fma0();
    fma1();
    fma2();
    fma3();
    fn0();
    fn1();
};
