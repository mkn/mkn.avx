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
#include "mkn/avx/unit.hpp"

#include "unit.hpp"

#include <array>
#include <optional>


namespace mkn::avx::detail
{
template<typename T, std::size_t N, std::size_t A = Options::ALIGN()>
struct _A_
{
    using arr_t = std::array<T, N>;

    alignas(A) arr_t arr;
};

} // namespace mkn::avx::detail


namespace mkn::avx
{

template<typename T, std::size_t N>
class Array : public detail::_A_<T, N>, public Unit<T, N>
{
    using This   = Array<T, N>;
    using Unit_t = Unit<T, N>;

public:
    using Arr = detail::_A_<T, N>;
    using Arr::arr;


    Array(std::nullopt_t const) // no default value!
        : Arr{}
        , Unit_t{arr.data(), arr.size()}
    {
    }

    Array(T const val = 0)
        : Arr{}
        , Unit_t{arr.data(), arr.size()}
    {
        **this = val;
    }

    Array(Array const& that)
        : Arr{}
        , Unit_t{arr.data(), arr.size()}
    {
        **this = *that;
    }
    Array(Array&& that)
        : Arr{}
        , Unit_t{arr.data(), arr.size()}
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
    auto inline operator+(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.add(*this, arr);
        return ret;
    }
    auto inline operator+(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.add(*this, make_span<N>(arr));
        return ret;
    }

    template<typename T0>
    auto inline operator-(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.sub(*this, arr);
        return ret;
    }
    auto inline operator-(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.sub(*this, make_span(arr));
        return ret;
    }

    template<typename T0>
    auto inline operator*(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.mul(*this, arr);
        return ret;
    }
    auto inline operator*(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.mul(*this, make_span(arr));
        return ret;
    }

    template<typename T0>
    auto inline operator/(Array<T0, N> const& arr) const
    {
        Array ret{};
        ret.div(*this, arr);
        return ret;
    }
    auto inline operator/(Arr::arr_t const& arr) const
    {
        Array ret{};
        ret.div(*this, make_span(arr));
        return ret;
    }


    Unit_t& super() { return *this; }
    Unit_t const& super() const { return *this; }
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
auto inline operator+(mkn::avx::Unit<T0, N> const& __restrict span,
                      mkn::avx::Unit<T1, N> const& __restrict arr)
{
    mkn::avx::Array<std::decay_t<T1>, N> ret{std::nullopt};
    *ret = span;
    ret += arr;
    return ret;
}

template<typename T0, typename T1, std::size_t N>
auto inline operator-(T0 const t0, mkn::avx::Array<T1, N> const& arr)
{
    return mkn::avx::Array<std::decay_t<T1>, N>{t0} - arr;
}


template<typename T0, typename T1, std::size_t N>
auto inline operator*(mkn::avx::Unit<T0, N> const& __restrict s0,
                      mkn::avx::Unit<T1, N> const& __restrict s1)
{
    mkn::avx::Array<std::decay_t<T1>, N> ret{std::nullopt};
    *ret = s0;
    ret *= s1;
    return ret;
}


#endif /* _MKN_AVX_ARRAY_HPP_ */
