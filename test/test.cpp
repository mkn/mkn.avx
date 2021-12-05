
#include "mkn/kul/log.hpp"
#include "mkn/avx.hpp"
#include "mkn/avx/vector.hpp"

#include <array>
#include <vector>
#include <cassert>
#include <iostream>

void vec()
{
    mkn::avx::Vector<double> a(101, 2), b(101, 3);
    a += b;
    a *= b;
    assert(a.data()[00] == 15);
    assert(a.data()[99] == 15);
    assert(a.data()[100] == 15);
}

void fma(){
    using AVX                         = mkn::avx::Type<double, 4>;
    static constexpr std::size_t SIZE = AVX::value_count * 3;
    static_assert((SIZE % AVX::value_count) == 0, "Bad AVX value_count for vector");

    std::vector<AVX> a(SIZE / AVX::value_count, 1);
    for (size_t i = 0; i < SIZE; i++)
        assert(a[0][i] == 1); // dangeresque

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

int main() noexcept
{
    std::cout << __FILE__ << std::endl;

    fma();
    vec();

    return 0;
}
