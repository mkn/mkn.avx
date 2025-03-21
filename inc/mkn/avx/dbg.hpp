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
#ifndef _MKN_AVX_DBG_HPP_
#define _MKN_AVX_DBG_HPP_

#include "mkn/avx/def.hpp"

#include <string>
#include <sstream>
#include <unordered_map>


namespace mkn::avx
{
struct Counter
{
    static auto& I()
    {
        static Counter i;
        return i;
    }

    void operator()(auto&&... args) { ++cnts[format(args...)]; }


    static auto format(auto&&... args)
    {
        std::stringstream ss;
        ((ss << args << ":"), ...);
        return ss.str();
    }

    std::unordered_map<std::string, std::size_t> cnts;
};

struct CountPoint
{
    CountPoint(auto&&... args) { Counter::I().cnts[Counter::format(args...)] = 0; }
};

} // namespace mkn::avx

#if defined(MKN_AVX_COUNT_FNS)
#define MKN_AVX_FN_COUNTER                                                                         \
    static mkn::avx::CountPoint __mkn_avx_count_point##__LINE__{__FILE__, __func__, __LINE__};     \
    mkn::avx::Counter::I()(__FILE__, __func__, __LINE__);

#else // !defined(MKN_AVX_FN_COUNTER)
#define MKN_AVX_FN_COUNTER

#endif // MKN_AVX_FN_COUNTER

#endif /* _MKN_AVX_DBG_HPP_ */
