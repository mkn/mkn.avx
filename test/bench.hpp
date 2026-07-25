
#ifndef _MKN_AVX_TEST_BENCH_HPP_
#define _MKN_AVX_TEST_BENCH_HPP_

#include "mkn/kul/time.hpp"



#if !defined(__INTEL_COMPILER) && !defined(__clang__) && defined(__GNUC__)
#if (__GNUC__ == 4 && __GNUC_MINOR__ > 3) || (__GNUC__ >= 5)
#define NO_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#else
#define NO_VECTORIZE _Pragma("GCC optimize(\"no-tree-vectorize\")")
#endif
#else
#define NO_VECTORIZE
#endif


struct Timer
{
    ~Timer()
    {
        std::cout << std::setprecision(20)
                  << "Time taken: " << static_cast<double>(now() - start) / div << std::endl;
    }

    std::uint64_t static now()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    double div                = 1;
    std::uint64_t const start = now();
};

#endif
