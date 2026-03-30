#include <vector>
#include <cstddef>
#include "mkn/avx.hpp"
#include "mkn/kul/assert.hpp"

#include "bench.hpp"

std::size_t constexpr static COUNT = 10000000;

namespace mkn::avx
{

template<typename T>
void test_unaligned()
{
    using Vec_t = std::vector<T>;

    std::vector<T> v0(COUNT, 1), v1(COUNT, 2);

    auto&& [a, b] = make_spans(v0, v1);

    {
        Timer timer{COUNT};
        b += a;
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
