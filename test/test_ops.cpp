// mkn::avx lane operations (fma, floor, abs, comparisons and masks, select, int32 conversions)
//  against the scalar std result per lane, at every width the build enables

#include "mkn/kul/assert.hpp"
#include "mkn/avx.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace mkn::avx
{

// awkward inputs: signed zeros, halves, integers, negatives, near integer
double constexpr static values[] = {-0.0,        0.0,     0.5,  -0.5,  1.0, -1.0, 2.25, -2.25,
                                    1.0 - 1e-15, -1e-300, 7.75, -7.75, 3.5, -3.5, 1e8,  -1e8};

std::size_t static checks = 0;

void check(bool const ok, char const* const what, std::size_t const lane)
{
    ++checks;
    if (!ok)
        std::cerr << "mkn::avx test_ops: " << what << " lane " << lane << std::endl;
    mkn::kul::abort_if_not(ok);
}

// the same bits (floor(-0.5) is -1, floor(-0.0) is -0.0: the sign matters)
bool same(double const a, double const b)
{ return a == b and std::signbit(a) == std::signbit(b); }

template<std::size_t N>
void ops(std::size_t const offset)
{
    using T = Type<double, N>;
    std::array<double, N> a, b, c, out;
    for (std::size_t l = 0; l < N; ++l)
    {
        a[l] = values[(offset + l) % std::size(values)];
        b[l] = values[(offset + 3 * l + 1) % std::size(values)];
        c[l] = values[(offset + 5 * l + 2) % std::size(values)];
    }
    T const va = unaligned_load<double, N>(a.data());
    T const vb = unaligned_load<double, N>(b.data());
    T const vc = unaligned_load<double, N>(c.data());

    unaligned_store(out.data(), fma(va, vb, vc));
    for (std::size_t l = 0; l < N; ++l)
        check(out[l] == std::fma(a[l], b[l], c[l]), "fma", l);

    unaligned_store(out.data(), floor(va));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], std::floor(a[l])), "floor", l);

    unaligned_store(out.data(), abs(va));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], std::abs(a[l])), "abs", l);

    unaligned_store(out.data(), select(lt(va, vb), va, vb));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], a[l] < b[l] ? a[l] : b[l]), "select lt", l);

    unaligned_store(out.data(), select(gt(va, vb), va, vb));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], a[l] > b[l] ? a[l] : b[l]), "select gt", l);

    auto const m_and = lt(va, vb) & gt(va, vc);
    auto const m_or  = lt(va, vb) | gt(va, vc);
    unaligned_store(out.data(), select(m_and, va, vc));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], (a[l] < b[l] and a[l] > c[l]) ? a[l] : c[l]), "mask and", l);
    unaligned_store(out.data(), select(m_or, va, vc));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], (a[l] < b[l] or a[l] > c[l]) ? a[l] : c[l]), "mask or", l);

    bool expect_any = false;
    for (std::size_t l = 0; l < N; ++l)
        expect_any = expect_any or a[l] < b[l];
    check(any(lt(va, vb)) == expect_any, "any", 0);
    check(!any(lt(va, va)), "any none", 0); // a < a nowhere

    // NaN: ordered comparisons are false
    T const nan = load<double, N>(std::numeric_limits<double>::quiet_NaN());
    check(!any(lt(nan, va) | gt(nan, va)), "NaN compare", 0);
}

// truncating conversions, and the int32 lanes' loads, stores and sums
template<std::size_t N>
void int32s(std::size_t const offset)
{
    std::array<double, N> a, back;
    std::array<std::int32_t, N> i, j;
    for (std::size_t l = 0; l < N; ++l)
        a[l] = values[(offset + l) % std::size(values)];
    auto const va = unaligned_load<double, N>(a.data());

    auto const vi = to_int32(va);
    unaligned_store(i.data(), vi);
    for (std::size_t l = 0; l < N; ++l)
        check(i[l] == static_cast<std::int32_t>(a[l]), "to_int32", l); // truncation

    unaligned_store(back.data(), to_double(vi));
    for (std::size_t l = 0; l < N; ++l)
        check(back[l] == static_cast<double>(i[l]), "to_double", l);

    // floor then convert: the cell of a position
    unaligned_store(i.data(), to_int32(floor(va)));
    for (std::size_t l = 0; l < N; ++l)
        check(i[l] == static_cast<std::int32_t>(std::floor(a[l])), "to_int32(floor)", l);

    // load(value) gives the register, not a Type
    auto const vj = unaligned_load<std::int32_t, N>(i.data())
                    + Type<std::int32_t, N>{load<std::int32_t, N>(3)};
    unaligned_store(j.data(), vj);
    for (std::size_t l = 0; l < N; ++l)
        check(j[l] == i[l] + 3, "int32 add", l);
}

template<std::size_t N>
void width(char const* const name)
{
    for (std::size_t offset = 0; offset < std::size(values); ++offset)
        ops<N>(offset);
    if constexpr (N > 2) // int32 lanes: 4 (AVX2), 8 (AVX-512F)
        for (std::size_t offset = 0; offset < std::size(values); ++offset)
            int32s<N>(offset);
    std::cout << "  " << name << " ok" << std::endl;
}

} // namespace mkn::avx

int main() noexcept
{
    std::cout << __FILE__ << std::endl;
    mkn::avx::width<1>("scalar");
#if defined(__SSE4_1__)
    mkn::avx::width<2>("double x2");
#endif
#if defined(__AVX2__) && defined(__FMA__)
    mkn::avx::width<4>("double x4");
#endif
#if defined(__AVX512F__)
    mkn::avx::width<8>("double x8");
#endif
    std::cout << mkn::avx::checks << " checks" << std::endl;
    return 0;
}
