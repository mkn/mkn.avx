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
#ifndef _MKN_AVX_SPAN_HPP_
#define _MKN_AVX_SPAN_HPP_

#include "mkn/avx/def.hpp"
#include "mkn/avx/types.hpp"

#include "mkn/kul/span.hpp"

#include <array>
#include <tuple>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>


namespace mkn::avx
{


template<typename T, std::size_t _N = Options::N<T>()>
class Span
{
protected:
    using R = std::decay_t<T>;

private:
    template<typename, std::size_t>
    friend class Span;

public:
    using value_type        = T;
    auto constexpr static N = Options::N<R>();
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

    Span(T* d, std::size_t const s) noexcept
        : u{d}
        , span{d, s}
    {
    }

public:
    inline Span(T* d) noexcept
        : u{d}
        , span{d, 1}
    {
    }
    inline Span(Span const& that)
        : u{that.u}
        , span{that.span}
    {
    }

    inline Span(Span&& that)
        : u{that.u}
        , span{that.span}
    {
    }


    template<typename T0, typename T1>
    void inline add(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] + v2[0];
    }


    template<typename T0, typename T1>
    void inline sub(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] - v2[0];
    }


    template<typename T0, typename T1>
    void inline mul(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] * v2[0];
    }


    template<typename T0, typename T1>
    void inline div(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        v0[0]                    = v1[0] / v2[0];
    }


    template<typename T0, typename T1, typename T2>
    void fma(Span<T0, N> const& a, Span<T1, N> const& b, Span<T2, N> const& c) noexcept
    {
        auto const& [v0, v1, v2, v3] = cast(*this, a, b, c);
        v0[0]                        = mkn::avx::fma(v1[0], v2[0], v3[0]);
    }



    template<typename T0>
    auto inline operator+=(Span<T0, N> const& that) noexcept
    {
        // auto const& [v0, v1] = cast(*this, that);
        // v0[0] += v1[0];
        **this += *that;
    }

    template<typename T0>
    auto inline operator+=(std::array<T0, N> const& arr) noexcept
    {
        Span<T0 const> that{arr};
        auto const& [v0, v1] = cast(*this, that);
        v0[0] += v1[0];
    }



    template<typename T0>
    auto inline operator-=(Span<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        v0[0] -= v1[0];
    }

    template<typename T0>
    auto inline operator-=(std::array<T0, N> const& arr) noexcept
    {
        Span<T0 const> that{arr};
        auto const& [v0, v1] = cast(*this, that);
        v0[0] -= v1[0];
    }

    template<typename T0>
    auto inline operator*=(Span<T0, N> const& that) noexcept
    {
        // auto const& [v0, v1] = cast(*this, that);
        // v0[0] *= v1[0];
        **this *= *that;
    }

    template<typename T0>
    auto inline operator*=(std::array<T0, N> const& arr) noexcept
    {
        Span<T0 const> that{arr};
        auto const& [v0, v1] = cast(*this, that);
        v0[0] *= v1[0];
    }

    auto& operator=(Span const& that) noexcept
    {
        auto const& [v1] = cast(that);
        store(data(), v1[0]);
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
    auto& operator=(Span<T0, N>&& that) = delete;


    template<typename T0>
    bool inline operator==(Span<T0, N> const& that) const noexcept
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

    auto data() const noexcept { return span.data(); }
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
    static inline auto& caster(Span<T0, N>& that) noexcept
    {
        static_assert(std::is_same_v<R, typename Span<T0, N>::R>);
        return that.avx();
    }

    template<typename T0>
    static inline auto& caster(Span<T0, N> const& that) noexcept
    {
        static_assert(std::is_same_v<R, typename Span<T0, N>::R>);
        return that.avx();
    }

    template<typename... Args>
    static inline auto cast(Args&&... args)
    {
        return std::forward_as_tuple(caster(args)...);
    }


    mkn::kul::Span<value_type> span;
};



template<typename T, std::size_t _N = Options::N<T>()>
class SpanSet : public Span<T, _N>
{
    using Super = Span<T, _N>;
    using R     = Super::R;
    using Super::span;

    template<typename, std::size_t>
    friend class SpanSet;
    using Super::cast;

public:
    using Super::data;
    using value_type        = Super::value_type;
    auto constexpr static N = Super::N;

    using AVX_t = Super::AVX_t;


    SpanSet(T* d, std::size_t const& s) noexcept
        : Super{d, s}
    {
    }

    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    SpanSet(C const& v) noexcept
        : Super{v.data(), v.size()}
    {
    }
    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    SpanSet(C& v) noexcept
        : Super{v.data(), v.size()}
    {
    }


    template<typename T0, typename T1>
    void inline add(SpanSet<T0, N> const& a, SpanSet<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] + v2[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] + b.span[i];
    }


    template<typename T0, typename T1>
    void inline sub(SpanSet<T0, N> const& a, SpanSet<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] - v2[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] - b.span[i];
    }



    template<typename T0, typename T1>
    void inline mul(SpanSet<T0, N> const& a, SpanSet<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] * v2[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] * b.span[i];
    }


    template<typename T0, typename T1>
    void inline div(SpanSet<T0, N> const& a, SpanSet<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] / v2[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] / b.span[i];
    }

    template<typename T0, typename T1, typename T2>
    void inline fma(SpanSet<T0, N> const& a, SpanSet<T1, N> const& b,
                    SpanSet<T2, N> const& c) noexcept
    {
        auto const& [v0, v1, v2, v3] = cast(*this, a, b, c);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = mkn::avx::fma(v1[i], v2[i], v3[i]);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] * b.span[i] + c.span[i];
    }



    template<typename T0>
    auto inline operator+=(SpanSet<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] += v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] += that.span[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator+=(Arr<T0, N> const& arr) noexcept
    {
        SpanSet<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] += v1[0];
    }
    auto inline operator+=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);
        (*this) += scratch;
    }

    template<typename T0>
    auto inline operator-=(SpanSet<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] -= v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] -= that[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator-=(Arr<T0, N> const& arr) noexcept
    {
        SpanSet<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] -= v1[0];
    }


    template<typename T0>
    auto inline operator*=(SpanSet<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] *= v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] *= that.span[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator*=(Arr<T0, N> const& arr) noexcept
    {
        SpanSet<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] *= v1[0];
    }
    auto inline operator*=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);
        (*this) *= scratch;
    }

    template<typename T0>
    auto inline operator/=(SpanSet<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] /= v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] /= that.span[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator/=(Arr<T0, N> const& arr) noexcept
    {
        SpanSet<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] /= v1[0];
    }
    auto inline operator/=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);
        (*this) /= scratch;
    }

    template<typename T0>
    auto& operator=(T0 const& that) noexcept
    {
        static_assert(std::is_same_v<R, std::decay_t<typename T0::value_type>>);
        std::memcpy(data(), that.data(), sizeof(T) * size());
        return *this;
    }

    template<typename T0>
    auto& operator=(SpanSet<T0, N>&& that) = delete;


    template<typename T0>
    bool inline operator==(SpanSet<T0, N> const& that) const noexcept
    {
        for (std::size_t i = 0; i < size(); ++i)
            if (span[i] != that.span[i])
                return false;
        return true;
    }

    bool inline operator==(T const t) const noexcept
    {
        for (std::size_t i = 0; i < size(); ++i)
            if (span[i] != t)
                return false;
        return true;
    }

    auto& size() const noexcept { return span.size(); }

    Super& super() { return *this; }
    Super const& super() const { return *this; }
    auto& operator*() { return super(); }
    auto& operator*() const { return super(); }

protected:
    auto inline modulo_leftover_idx(auto const siz) { return siz - siz % N; }
    auto inline modulo_leftover_idx() { return modulo_leftover_idx(size()); }


private:
    alignas(Options::ALIGN()) std::array<T, N> scratch{};
};


template<std::size_t N, typename Container>
auto inline make_span(Container& container, auto const start = 0) noexcept
{
    if constexpr (mkn::kul::is_span_like_v<Container>)
        return Span<typename Container::value_type, N>{container.data() + start};
    else
        return Span<std::remove_pointer_t<Container>, N>{container + start};
}
template<std::size_t N, typename Container>
auto inline make_span(Container const& container, auto const start = 0) noexcept
{
    if constexpr (mkn::kul::is_span_like_v<Container>)
        return Span<typename Container::value_type const, N>{container.data() + start};
    else
        return Span<std::remove_pointer_t<Container> const, N>{container + start};
}

template<typename Container>
auto make_span(Container& container) noexcept
{
    return SpanSet<typename Container::value_type>{container};
}
template<typename Container>
auto make_span(Container const& container) noexcept
{
    return SpanSet<typename Container::value_type const>{container};
}

template<typename Container>
auto make_span(Container& container, auto const start, auto const size) noexcept
{
    return SpanSet<typename Container::value_type>{container.data() + start, size};
}
template<typename Container>
auto make_span(Container const& container, auto const start, auto const size) noexcept
{
    return SpanSet<typename Container::value_type const>{container.data() + start, size};
}

template<std::size_t N, typename... Containers>
auto inline make_spans(std::size_t start, Containers&&... containers)
{
    return std::make_tuple(make_span<N>(containers.data(), start)...);
}


template<std::size_t N, typename... Containers>
auto inline make_spans(Containers&&... containers)
{
    return std::make_tuple(make_span<N>(containers.data())...);
}


template<typename... Containers>
auto inline make_spans(Containers&&... containers)
{
    return std::make_tuple(make_span(containers)...);
}


} // namespace mkn::avx

#endif /* _MKN_AVX_SPAN_HPP_ */
