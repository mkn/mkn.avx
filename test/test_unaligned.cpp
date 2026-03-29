
#include "mkn/kul/dbg.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/assert.hpp"

#include "mkn/avx.hpp"
#include "mkn/avx/span.hpp"
#include "mkn/avx/vector.hpp"

#include "bench.hpp"

#include <cmath>
#include <iostream>

std::size_t constexpr static COUNT = 1000000;

namespace mkn::avx
{

template<typename T>
void test_unaligned()
{
    MKN_KUL_DBG_FUNC_ENTER;


    using Vec_t = std::vector<T>;

    std::vector<T> v0(COUNT, 1), v1(COUNT, 2);

    static_assert(is_aligned<Vec_t>() == false);

    {
        Timer timer{COUNT};
        auto&& [a, b] = make_spans(v0, v1);
        b += a;

        static_assert(decltype(a)::N >= 2);
    }

    mkn::kul::abort_if_not(v1.front() == 3);
    mkn::kul::abort_if_not(v1.back() == 3);
}

} // namespace mkn::avx

int main() noexcept
{
    std::cout << __FILE__ << std::endl;

    mkn::avx::test_unaligned<double>();

    return 0;
}
