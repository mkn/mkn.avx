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
#ifndef _MKN_AVX_ARRAY_HPP_
#define _MKN_AVX_ARRAY_HPP_

#include "mkn/avx/def.hpp"
#include "mkn/avx/span.hpp"

#include "span.hpp"

#include <array>
#include <cstdint>
#include <optional>


namespace mkn::avx::detail
{
template<typename T, std::size_t N, std::size_t A = Options::ALIGN()>
struct _A_
{
    using arr_t = std::array<T, N>;

    // _A_() {}
    // _A_(T const val = 0) { store(float); }

    alignas(A) arr_t arr;
};

} // namespace mkn::avx::detail


namespace mkn::avx
{

template<typename T, std::size_t N>
class Array : public detail::_A_<T, N>, public Span<T, N>
{
    using This   = Array<T, N>;
    using Span_t = Span<T, N>;

public:
    using Arr = detail::_A_<T, N>;
    using Arr::arr;


    Array(std::nullopt_t const) // no default value!
        : Arr{}
        , Span_t{arr.data(), arr.size()}
    {
    }

    Array(T const val = 0)
        : Arr{}
        , Span_t{arr.data(), arr.size()}
    {
        **this = val;
    }

    Array(Array const& that)
        : Arr{}
        , Span_t{arr.data(), arr.size()}
    {
        **this = *that;
    }
    Array(Array&& that)
        : Arr{}
        , Span_t{arr.data(), arr.size()}
    {
        **this = *that;
    }

    Array& operator=(Array const& that)
    {
        **this = *that;
        return *this;
    };
    // Array& operator=(T const& that)
    // {
    //     arr.fill(that);
    //     return *this;
    // };

    Array& operator=(Array&& that)
    {
        **this = *that;
        return *this;
    };

    // Array& operator=(Array&& that) = delete;


    template<typename T0>
    auto operator+(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.add(*this, arr);
        return ret;
    }
    auto operator+(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.add(*this, make_span<N>(arr));
        return ret;
    }

    template<typename T0>
    auto operator-(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.sub(*this, arr);
        return ret;
    }
    auto operator-(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.sub(*this, make_span(arr));
        return ret;
    }

    template<typename T0>
    auto operator*(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.mul(*this, arr);
        return ret;
    }
    auto operator*(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.mul(*this, make_span(arr));
        return ret;
    }

    template<typename T0>
    auto operator/(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.div(*this, arr);
        return ret;
    }
    auto operator/(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.div(*this, make_span(arr));
        return ret;
    }


    Span_t& super() { return *this; }
    Span_t const& super() const { return *this; }
    auto& operator*() { return super(); }
    auto& operator*() const { return super(); }

    auto begin() { return arr.begin(); }
    auto begin() const { return arr.begin(); }
    auto end() { return arr.end(); }
    auto end() const { return arr.end(); }

    auto data() { return arr.data(); }
    auto data() const { return arr.data(); }
    auto constexpr static size() { return N; }

    template<typename Ret = This, typename Fn, typename Arr>
    auto static FROM(Fn const& fn, Arr const& arr)
    {
        Ret ret;
        for (std::size_t i = 0; i < N; ++i)
            ret[i] = fn(arr[i]);
        return ret;
    }
};



} /* namespace mkn::avx */



template<typename T0, typename T1, std::size_t N>
auto operator+(mkn::avx::Span<T0> const& span, mkn::avx::Array<T1, N> const& arr)
{
    mkn::avx::Array<std::decay_t<T1>, N> ret{};
    ret.add(span, arr);
    return ret;
}

template<typename T0, typename T1, std::size_t N>
auto operator-(T0 const t0, mkn::avx::Array<T1, N> const& arr)
{
    return mkn::avx::Array<std::decay_t<T1>, N>{t0} - arr;
}


#endif /* _MKN_AVX_ARRAY_HPP_ */
