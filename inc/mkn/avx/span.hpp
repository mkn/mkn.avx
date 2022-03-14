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
    using R = std::decay_t<T>;

public:
    using value_type        = T;
    auto constexpr static N = Options::N<R>();
    // auto static constexpr is_const = std::is_const_v<T>;

    using AVX_t = mkn::avx::Type<T, N>;

    Span(T* d, std::size_t s) noexcept
        : span{d, s}
    {
    }

    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    Span(C const& v) noexcept
        : span{v.data(), v.size()}
    {
    }
    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    Span(C& v) noexcept
        : span{v.data(), v.size()}
    {
    }


    template<typename T0, typename T1>
    void add(Span<T0> const& a, Span<T1> const& b) noexcept
    {
        auto& v0 = caster(*this);
        auto& v1 = caster(a);
        auto& v2 = caster(b);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] + v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] = a.span[i] + b.span[i];
    }

    template<typename T0, typename T1>
    void mul(Span<T0> const& a, Span<T1> const& b) noexcept
    {
        static_assert(sizeof(mkn::kul::Span<AVX_t>) == sizeof(mkn::kul::Span<T>));

        auto& v0 = caster(*this);
        auto& v1 = caster(a);
        auto& v2 = caster(b);

        assert(v0.size() == v1.size() and v1.size() == v2.size());
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] * v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] = a.span[i] * b.span[i];
    }


    template<typename T0>
    void operator+=(Span<T0> const& that) noexcept
    {
        assert(this->size() >= that.size());

        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] += v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] += that.span[i];
    }

    template<typename T0>
    void operator*=(Span<T0> const& that) noexcept
    {
        assert(this->size() >= that.size());

        static_assert(sizeof(mkn::kul::Span<AVX_t>) == sizeof(mkn::kul::Span<T>));

        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] *= v1[i];
        for (std::size_t i = size() - size() % N; i < size(); ++i)
            span[i] *= that.span[i];
    }

    template<typename T0>
    bool operator==(Span<T0> const& that) const noexcept
    { // TODO vectorize
        for (std::size_t i = 0; i < size(); ++i)
            if (span[i] != that.span[i])
                return false;
        return true;
    }

    bool operator==(T const t) const noexcept
    { // TODO vectorize
        for (std::size_t i = 0; i < size(); ++i)
            if (span[i] != t)
                return false;
        return true;
    }

    auto& size() const noexcept { return span.size(); }
    auto data() const noexcept { return span.data(); }
    auto data() noexcept { return span.data(); }

    auto& operator[](std::size_t i) const noexcept { return span[i]; }
    auto& operator[](std::size_t i) noexcept { return span[i]; }

    auto& operator()() noexcept { return span; }
    auto& operator()() const noexcept { return span; }


    mkn::kul::Span<value_type> span;

private:
    template<typename T0, typename = std::enable_if_t<std::is_same_v<R, std::decay_t<T0>>>>
    static auto& caster(Span<T0>& that) noexcept
    {
        return *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&that.span);
    }

    template<typename T0, typename = std::enable_if_t<std::is_same_v<R, std::decay_t<T0>>>>
    static auto& caster(Span<T0> const& that) noexcept
    {
        return *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&that.span);
    }
};



template<typename Container>
auto make_span(Container& container) noexcept
{
    return Span<typename Container::value_type>{container};
}
template<typename Container>
auto make_span(Container const& container) noexcept
{
    return Span<typename Container::value_type const>{container};
}


} /* namespace mkn::avx */

#endif /* _MKN_AVX_SPAN_HPP_ */
