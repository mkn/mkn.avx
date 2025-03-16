/**
Copyright (c) 2024, Philip Deegan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Philip Deegan nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _MKN_AVX_DEF_HPP_
#define _MKN_AVX_DEF_HPP_


#include <cassert>
#include <cstdint>
#include <immintrin.h> // avx
#include <type_traits>

// try running
//   clang -dM -march=native -E - < /dev/null | grep AVX
//
#if defined(__AVX__) && !defined(MKN_AVX_1_ACTIVE)
#define MKN_AVX_1_ACTIVE 1
#endif

#if !defined(MKN_AVX_1_ACTIVE)
#define MKN_AVX_1_ACTIVE 0
#endif


#if defined(__AVX2__) && !defined(MKN_AVX_2_ACTIVE)
#define MKN_AVX_2_ACTIVE 1
#endif

#if !defined(MKN_AVX_2_ACTIVE)
#define MKN_AVX_2_ACTIVE 0
#endif


#if defined(__AVX512F__) && !defined(MKN_AVX_512_ACTIVE)
#define MKN_AVX_512_ACTIVE 1
#endif

#if !defined(MKN_AVX_512_ACTIVE)
#define MKN_AVX_512_ACTIVE 0
#endif


#if !defined(MKN_AVX_ALIGN_AS)
#define MKN_AVX_ALIGN_AS 32
#endif

namespace mkn::avx
{
struct Options
{
    bool static constexpr AVX               = MKN_AVX_1_ACTIVE;
    bool static constexpr AVX2              = MKN_AVX_2_ACTIVE;
    bool static constexpr AVX512            = MKN_AVX_512_ACTIVE;
    std::uint16_t static constexpr ALIGN_AS = MKN_AVX_ALIGN_AS;

    template<typename AT, std::uint16_t operands = 1>
    std::uint16_t static constexpr N()
    {
        using T = std::decay_t<AT>;
        if constexpr (AVX512)
            return 512 / 8 / sizeof(T) / operands;
        else if constexpr (AVX2)
            return 256 / 8 / sizeof(T) / operands;
        else if constexpr (AVX)
            return 128 / 8 / sizeof(T) / operands;
        else
            return 1;
    }
};


} /* namespace mkn::avx */

#endif /* _MKN_AVX_DEF_HPP_ */
