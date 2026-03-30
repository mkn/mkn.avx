
#include "mkn/kul/assert.hpp"

#include "mkn/avx/vector.hpp"

#include "betterbench.hpp"

#include <stdexcept>

using base_type = double;

std::size_t constexpr size = 10000000;
std::size_t constexpr arrs = size / 4;

int main(/*int argc, char** argv*/)
{
    std::cout << __FILE__ << std::endl;
    using T     = __m256d;
    using AVX   = mkn::avx::Type<base_type, 4>;
    using AVX_t = mkn::avx::Type_<base_type, 4>;

    base_type b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    op(b0, b1, b2, b3);

    mkn::avx::Vector<AVX> d0(arrs), d1(arrs), d2(arrs), d3(arrs);

    auto const prefetcher
        = [](auto i, auto&... args) { ((__builtin_prefetch(args.data() + i, 1, 2)), ...); };

    auto const op = [](auto&... args) {
        auto&& [a, b, c, d] = std::forward_as_tuple(args...);

        T const t0 = AVX_t::set_v(2);

        a = AVX_t::set_v(10);
        c = AVX_t::set_v(400);

        T const t1 = AVX_t::set_v(2);
        T const t3 = AVX_t::mul(t0, t1);
        b          = AVX_t::add(b, t3);

        c    = AVX_t::add(c, t1);
        T t4 = AVX_t::set_v(100);
        t4   = AVX_t::mul(t4, t3); // t4 *= t3;
        d    = AVX_t::add(a, AVX_t::mul(b, c));
        d    = AVX_t::add(d, t4); // d += t4;
        d    = AVX_t::mul(d, a);  // d *= a;
        d    = AVX_t::mul(d, b);  // d *= b;
        d    = AVX_t::mul(d, c);  // d *= c;
    };

    std::size_t i = 0;
    {
        for (; i < arrs; ++i)
            op(d0[i](), d1[i](), d2[i](), d3[i]());
    }

    mkn::kul::abort_if_not(d3[arrs - 1][3] == b3);
};
