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
#ifndef _MKN_AVX_UNIT_HPP_
#define _MKN_AVX_UNIT_HPP_

#include "mkn/kul/span.hpp"

#include "mkn/avx/def.hpp"
#include "mkn/avx/types.hpp"

#include <array>
#include <tuple>
#include <cassert>
#include <cstddef>
#include <cstring>


namespace mkn::avx
{

template<typename T, std::size_t _N = Options::N<std::decay_t<T>>()>
class Unit
{
protected:
    using R = std::decay_t<T>;

private:
    template<typename, std::size_t>
    friend class Unit;

public:
    using value_type        = T;
    auto constexpr static N = _N;
    using AVX_t             = mkn::avx::Type<R, N>;

protected:
    union U
    {
        T* arr;
        mkn::avx::Type<R, N>* i;

        auto data() { return arr; }
        auto data() const { return arr; }
        auto constexpr static size() { return N; }
    };

    U u;

    auto& avx() { return u.i; }
    auto& avx() const { return u.i; }

    Unit(T* d, std::size_t const s) noexcept
        : u{d}
        , span{d, s}
    {
    }

public:
    inline Unit(T* d) noexcept
        : u{d}
        , span{d, 1}
    {
    }
    inline Unit(Unit const& that)
        : u{that.u}
        , span{that.span}
    {
    }

    inline Unit(Unit&& that)
        : u{that.u}
        , span{that.span}
    {
    }


    template<typename T0, typename T1>
    void inline add(Unit<T0, N> const& a, Unit<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] + v2[0];
    }


    template<typename T0, typename T1>
    void inline sub(Unit<T0, N> const& a, Unit<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] - v2[0];
    }


    template<typename T0, typename T1>
    void inline mul(Unit<T0, N> const& a, Unit<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] * v2[0];
    }


    template<typename T0, typename T1>
    void inline div(Unit<T0, N> const& a, Unit<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] / v2[0];
    }


    template<typename T0, typename T1, typename T2>
    void fma(Unit<T0, N> const& a, Unit<T1, N> const& b, Unit<T2, N> const& c) noexcept
    {
        auto const& [v0, v1, v2, v3] = cast(*this, a, b, c);
        v0[0]                        = mkn::avx::fma(v1[0], v2[0], v3[0]);
    }



    template<typename T0>
    auto inline operator+=(Unit<T0, N> const& that) noexcept
    {
        // auto const& [v0, v1] = cast(*this, that);
        // v0[0] += v1[0];
        **this += *that;
    }

    template<typename T0>
    auto inline operator+=(std::array<T0, N> const& arr) noexcept
    {
        Unit<T0 const> that{arr};
        auto const& [v0, v1] = cast(*this, that);
        v0[0] += v1[0];
    }



    template<typename T0>
    auto inline operator-=(Unit<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        v0[0] -= v1[0];
    }

    template<typename T0>
    auto inline operator-=(std::array<T0, N> const& arr) noexcept
    {
        Unit<T0 const> that{arr};
        auto const& [v0, v1] = cast(*this, that);
        v0[0] -= v1[0];
    }

    template<typename T0>
    auto inline operator*=(Unit<T0, N> const& that) noexcept
    {
        // auto const& [v0, v1] = cast(*this, that);
        // v0[0] *= v1[0];
        **this *= *that;
    }

    template<typename T0>
    auto inline operator*=(std::array<T0, N> const& arr) noexcept
    {
        Unit<T0 const> that{arr};
        auto const& [v0, v1] = cast(*this, that);
        v0[0] *= v1[0];
    }

    auto& operator=(Unit const& that) noexcept
    {
        store(data(), that.avx()[0]);
        return *this;
    }

    auto& operator=(T const& v) noexcept
    {
        auto const& [v0] = cast(*this);
        store(v0[0], v);
        return *this;
    }

    template<typename T0>
    auto& operator=(T0 const& that) noexcept
    {
        if constexpr (mkn::kul::is_span_like_v<T0>)
        {
            auto const& [v1] = cast(that);
            store(data(), v1[0]);
        }
        else
        {
            *this = static_cast<T>(that);
        }

        return *this;
    }

    template<typename T0>
    auto& operator=(Unit<T0, N>&& that) = delete;


    template<typename T0>
    bool inline operator==(Unit<T0, N> const& that) const noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
            if (span[i] != that.span[i])
                return false;
        return true;
    }

    bool inline operator==(T const t) const noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
            if (span[i] != t)
                return false;
        return true;
    }

    // auto data() const noexcept { return span.data(); }
    auto data() noexcept { return span.data(); }
    auto static constexpr size() noexcept { return N; }

    auto& operator[](std::size_t i) const noexcept { return span[i]; }
    auto& operator[](std::size_t i) noexcept { return span[i]; }

    auto& operator()() noexcept { return span; }
    auto& operator()() const noexcept { return span; }

protected:
    auto& operator*() { return *avx(); }
    auto& operator*() const { return *avx(); }

    template<typename T0>
    static inline auto& caster(Unit<T0, N>& that) noexcept
    {
        static_assert(std::is_same_v<R, typename Unit<T0, N>::R>);
        return that.avx();
    }

    template<typename T0>
    static inline auto& caster(Unit<T0, N> const& that) noexcept
    {
        static_assert(std::is_same_v<R, typename Unit<T0, N>::R>);
        return that.avx();
    }

    template<typename... Args>
    static inline auto cast(Args&&... args)
    {
        return std::forward_as_tuple(caster(args)...);
    }


    mkn::kul::Span<value_type> span;
};



} // namespace mkn::avx

#endif /* _MKN_AVX_UNIT_HPP_ */
