/**
Copyright (c) 2020, Philip Deegan.
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


// #define __MMX_WITH_SSE__ 1
// #define __SSE2_MATH__ 1
// #define __SSE__ 1
// #define __SSE2__ 1
// #define __SSE_MATH__ 1


#if !defined(__AVX__)
#pragma message("__AVX__ not defined")
#define __AVX__ 0
#endif


#if !defined(__AVX2__)
#pragma message("__AVX2__ not defined")
#define __AVX2__ 0
#endif



namespace mkn::avx
{
struct Options
{
    bool static constexpr AVX  = __AVX__;
    bool static constexpr AVX2 = __AVX2__;

    template<typename T, std::uint16_t operands = 2>
    auto static constexpr N()
    {
        if constexpr (AVX2)
            return 256 / 8 / sizeof(T) / operands;
    }
};


} /* namespace mkn::avx */

#endif /* _MKN_AVX_DEF_HPP_ */
