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
#ifndef _MKN_AVX_SPAN_HPP_
#define _MKN_AVX_SPAN_HPP_


#include <array>
#include <vector>
#include <cassert>
#include <cstdint>

#include "mkn/kul/span.hpp"

#include "mkn/avx/def.hpp"
#include "mkn/avx/types.hpp"


namespace mkn::avx
{
template<typename T>
class Span
{
public:
    auto constexpr static N = Options::N<T>();

    using AVX_t = mkn::avx::Type<T, N>;

    Span(T* d, std::size_t s)
        : span{d, s}
    {
    }

    Span(std::vector<T>& v)
        : span{v.data(), v.size()}
    {
    }

    void add(Span const& a, Span const& b)
    {
        auto& v0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&this->span);
        auto& v1 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&a.span);
        auto& v2 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&b.span);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] + v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] = a.span[i] + b.span[i];
    }

    void mul(Span const& a, Span const& b)
    {
        auto& v0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&this->span);
        auto& v1 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&a.span);
        auto& v2 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&b.span);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] * v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] = a.span[i] * b.span[i];
    }

    void operator+=(Span const& that)
    {
        assert(this->size() >= that.size());

        auto& v0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&this->span);
        auto& v1 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&that.span);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] += v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] += that.span[i];
    }

    void operator*=(Span const& that)
    {
        assert(this->size() >= that.size());

        auto& v0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&this->span);
        auto& v1 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&that.span);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] *= v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] *= that.span[i];
    }

    auto& size() const { return span.size(); }
    auto data() const { return span.data(); }
    auto data() { return span.data(); }

    auto& operator[](std::size_t i) const { return span[i]; }
    auto& operator[](std::size_t i) { return span[i]; }

    auto& operator()() { return span; }
    auto& operator()() const { return span; }

private:
    template<typename T0>
    auto& cast_to_AVX(T0 t)
    {
        return *reinterpret_cast<mkn::avx::Type<T, N>*>(cast_to_N(t));
    }

    template<typename T0>
    auto cast_to_N(T0 t)
    {
        return reinterpret_cast<std::array<T, N>*>(t);
    }

    mkn::kul::Span<T> span;
};

} /* namespace mkn::avx */

#endif /* _MKN_AVX_SPAN_HPP_ */
