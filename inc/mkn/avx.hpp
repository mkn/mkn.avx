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
#ifndef _MKN_AVX_HPP_
#define _MKN_AVX_HPP_

#include "mkn/avx/span.hpp"
#include "mkn/avx/vector.hpp"


#if 0 // sample code

#include "mkn/avx.hpp"
#include "mkn/kul/log.hpp"
#include <cstdlib> // for abort

int main(){

    std::size_t constexpr SIZE = 1e6;

    mkn::avx::Vector<float> v0(SIZE, 1), v1(SIZE, 2);
    auto&& [a, b] = mkn::avx::make_spans(v0, v1);

    a += b;

    for(std::size_t i = 0; i < SIZE; ++i)
        if(a[i] != 3) std::abort();

    KOUT(NON) << __FILE__;
    return 0;
}
#endif             // sample code


namespace mkn::avx
{
template<std::size_t N, typename Data>
auto inline make_span(Data* data, auto const start = 0) noexcept
{
    return Unit<Data, N>{data + start};
}

template<std::size_t N, typename Container>
auto inline make_span(Container& container, auto const start = 0) noexcept
{
    return Unit<typename Container::value_type, N>{container.data() + start};
}

template<typename Container>
auto make_span(Container& container) noexcept
{
    using vt        = Container::value_type;
    using real_type = std::conditional_t<std::is_const_v<Container>, vt const, vt>;
    if constexpr (is_aligned<Container>())
        return Span<real_type>{container};
    else
        return UnSpan<real_type>{container};
}

template<typename Container>
auto make_span(Container& container, auto const start, auto const size) noexcept
{
    return Span<typename Container::value_type>{container.data() + start, size};
}

template<std::size_t N, typename... Containers>
auto inline make_spans(std::size_t start, Containers&&... containers)
{
    return std::make_tuple(make_span<N>(containers.data(), start)...);
}


template<typename... Containers>
auto inline make_spans(Containers&&... containers)
{
    return std::make_tuple(make_span(containers)...);
}



} // namespace mkn::avx




#endif /* _MKN_AVX_HPP_ */
