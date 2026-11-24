// mkn::avx lane operations (fma, floor, abs, comparisons and masks, select, int32 conversions)
//  against the scalar std result per lane, at every width the build enables

#include "mkn/kul/test.hpp"
#include "mkn/avx.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>

MKN_KUL_TEST_DECLARE();

namespace mkn::avx
{

// awkward inputs: signed zeros, halves, integers, negatives, near integer
double constexpr static values[] = {-0.0,        0.0,     0.5,  -0.5,  1.0, -1.0, 2.25, -2.25,
                                    1.0 - 1e-15, -1e-300, 7.75, -7.75, 3.5, -3.5, 1e8,  -1e8};

// per operation, over every lane and offset of the width under test: one check each (width())
std::map<std::string, bool> static results;

void check(bool const ok, char const* const what, std::size_t const lane)
{
    if (!ok)
        std::cerr << "mkn::avx test_ops: " << what << " lane " << lane << std::endl;
    auto const [it, fresh] = results.try_emplace(what, ok);
    if (!fresh)
        it->second = it->second and ok;
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

// masked loads and stores: false lanes read as 0 and leave memory alone, even past the end of an
//  allocation of the true lanes only (a partial block at the end of an array)
template<std::size_t N>
void masked(std::size_t const offset)
{
    using T = Type<double, N>;
    std::array<double, N> a, b, out;
    for (std::size_t l = 0; l < N; ++l)
    {
        a[l] = values[(offset + l) % std::size(values)];
        b[l] = values[(offset + 3 * l + 1) % std::size(values)];
    }
    auto const va = unaligned_load<double, N>(a.data());
    auto const vb = unaligned_load<double, N>(b.data());
    auto const m  = lt(va, vb);

    double constexpr sentinel = 12345.5;
    out.fill(sentinel);
    mask_store(m, out.data(), va);
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], a[l] < b[l] ? a[l] : sentinel), "mask_store", l);

    unaligned_store(out.data(), mask_load(m, a.data()));
    for (std::size_t l = 0; l < N; ++l)
        check(same(out[l], a[l] < b[l] ? a[l] : 0.), "mask_load", l);

    // the first k lanes only, on exactly k doubles
    for (std::size_t k = 1; k <= N; ++k)
    {
        std::array<double, N> idx;
        for (std::size_t l = 0; l < N; ++l)
            idx[l] = static_cast<double>(l);
        auto const first_k
            = lt(unaligned_load<double, N>(idx.data()), T{load<double, N>(static_cast<double>(k))});
        auto* const tail = new double[k];
        for (std::size_t l = 0; l < k; ++l)
            tail[l] = sentinel;
        mask_store(first_k, tail, va);
        for (std::size_t l = 0; l < k; ++l)
            check(same(tail[l], a[l]), "mask_store first k", l);
        unaligned_store(out.data(), mask_load(first_k, tail));
        for (std::size_t l = 0; l < N; ++l)
            check(same(out[l], l < k ? a[l] : 0.), "mask_load first k", l);
        delete[] tail;
    }
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
    {
        ops<N>(offset);
        masked<N>(offset);
    }
    if constexpr (N > 2) // int32 lanes: 4 (AVX2), 8 (AVX-512F)
        for (std::size_t offset = 0; offset < std::size(values); ++offset)
            int32s<N>(offset);
    for (auto const& [what, ok] : results)
        MKN_KUL_TEST_CHECK(ok, std::string{name} + " " + what);
    results.clear();
}

} // namespace mkn::avx

int main() noexcept
{
    std::cout << __FILE__ << std::endl;
    mkn::avx::width<1>("scalar");
// the widths mkn::avx enables (def.hpp): AVX implies SSE4.1, AVX2 here comes with FMA
#if MKN_AVX_1_ACTIVE
    mkn::avx::width<2>("double x2");
#endif
#if MKN_AVX_2_ACTIVE
    mkn::avx::width<4>("double x4");
#endif
#if MKN_AVX_512_ACTIVE
    mkn::avx::width<8>("double x8");
#endif
    return 0;
}
