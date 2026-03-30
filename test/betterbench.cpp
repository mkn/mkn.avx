
#include "mkn/kul/assert.hpp"

#include "betterbench.hpp"

#include <stdexcept>

using base_type = double;

#if defined(MKN_AVX_BENCH_AVX_N)
#define MKN_AVX_BENCH_TYPE mkn::avx::Array<base_type, mkn::avx::Options::N<base_type>()>
#else
#define MKN_AVX_BENCH_TYPE base_type
#endif


std::size_t constexpr size = 10000000;


int main(/*int argc, char** argv*/)
{
    std::cout << __FILE__ << std::endl;

    base_type b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    op(b0, b1, b2, b3);

    {
        SoA<MKN_AVX_BENCH_TYPE, base_type> soa{size};
        assert(soa);
        soa();
        mkn::kul::abort_if_not(soa.v3.data()[soa.v3.size() - 1] == b3);
    }
    {
        AoS<base_type> aos{size};
        assert(aos);
        aos();
        mkn::kul::abort_if_not(aos.data()[aos.size() - 1].v3 == b3);
    }
};
