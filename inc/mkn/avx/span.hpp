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

#include "mkn/kul/span.hpp"

#include "mkn/avx/def.hpp"
#include "mkn/avx/unit.hpp"
#include "mkn/avx/types.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>


namespace mkn::avx
{

// contract: size() is an exact multiple of N - no remainder/tail handling.
// callers that cannot guarantee this must use AsymmetricSpan instead.
template<typename T, std::size_t _N = Options::N<T>()>
class Span : public Unit<T, _N>
{
    using Super = Unit<T, _N>;
    using R     = Super::R;

    template<typename, std::size_t>
    friend class Span;
    template<typename, std::size_t>
    friend class AsymmetricSpan;

protected:
    using Super::cast;
    using Super::span;

public:
    using Super::data;
    using value_type        = Super::value_type;
    auto constexpr static N = Super::N;

    using AVX_t = Super::AVX_t;


    Span(T* d, std::size_t const& s) noexcept
        : Super{d, s}
    {
    }

    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    Span(C const& v) noexcept
        : Super{v.data(), v.size()}
    {
    }
    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    Span(C& v) noexcept
        : Super{v.data(), v.size()}
    {
    }


    template<typename T0, typename T1>
    void inline add(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] = v1[i] + v2[i];
    }


    template<typename T0, typename T1>
    void inline sub(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] = v1[i] - v2[i];
    }



    template<typename T0, typename T1>
    void inline mul(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] = v1[i] * v2[i];
    }


    template<typename T0, typename T1>
    void inline div(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto const& [v0, v1, v2] = cast(*this, a, b);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] = v1[i] / v2[i];
    }

    template<typename T0, typename T1, typename T2>
    void inline fma(Span<T0, N> const& a, Span<T1, N> const& b, Span<T2, N> const& c) noexcept
    {
        auto const& [v0, v1, v2, v3] = cast(*this, a, b, c);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] = mkn::avx::fma(v1[i], v2[i], v3[i]);
    }



    template<typename T0>
    auto inline operator+=(Span<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] += v1[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator+=(Arr<T0, N> const& arr) noexcept
    {
        Span<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] += v1[0];
    }
    auto inline operator+=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);
        (*this) += scratch;
    }

    template<typename T0>
    auto inline operator-=(Span<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] -= v1[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator-=(Arr<T0, N> const& arr) noexcept
    {
        Span<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] -= v1[0];
    }


    template<typename T0>
    auto inline operator*=(Span<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] *= v1[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator*=(Arr<T0, N> const& arr) noexcept
    {
        Span<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] *= v1[0];
    }
    auto inline operator*=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);
        (*this) *= scratch;
    }

    template<typename T0>
    auto inline operator/=(Span<T0, N> const& that) noexcept
    {
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
            v0[i] /= v1[i];
    }
    template<template<typename, std::size_t> typename Arr, typename T0>
    auto inline operator/=(Arr<T0, N> const& arr) noexcept
    {
        Span<T0 const, N> const that{arr};
        auto const& [v0, v1] = cast(*this, that);
        for (std::size_t i = 0; i < batches(); ++i)
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
    auto& operator=(Span<T0, N>&& that) = delete;


    template<typename T0>
    bool inline operator==(Span<T0, N> const& that) const noexcept
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
    std::size_t batches() const { return size() / N; }

private:
    alignas(Options::ALIGN()) std::array<T, N> scratch{};
};


// arbitrary/unknown size - not guaranteed to be an exact multiple of N, so
// every op runs Span's batched loop and then mops up the ragged tail with a
// scalar pass over [modulo_leftover_idx(), size())
template<typename T, std::size_t _N = Options::N<T>()>
class AsymmetricSpan : public Span<T, _N>
{
    using Super = Span<T, _N>;
    using R     = Super::R;

    template<typename, std::size_t>
    friend class AsymmetricSpan;

protected:
    using Super::cast;
    using Super::span;
    using Super::batches;

public:
    using Super::Super;

    using Super::data;
    using value_type        = Super::value_type;
    auto constexpr static N = Super::N;

    using AVX_t = Super::AVX_t;

    using Super::add;
    using Super::sub;
    using Super::mul;
    using Super::div;
    using Super::fma;
    using Super::operator+=;
    using Super::operator-=;
    using Super::operator*=;
    using Super::operator/=;
    using Super::operator=;
    using Super::operator==;
    using Super::size;

    template<typename T0, typename T1>
    void inline add(AsymmetricSpan<T0, N> const& a, AsymmetricSpan<T1, N> const& b) noexcept
    {
        Span<T0, N> const& sa = a;
        Span<T1, N> const& sb = b;
        Super::add(sa, sb);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] + b.span[i];
    }

    template<typename T0, typename T1>
    void inline sub(AsymmetricSpan<T0, N> const& a, AsymmetricSpan<T1, N> const& b) noexcept
    {
        Span<T0, N> const& sa = a;
        Span<T1, N> const& sb = b;
        Super::sub(sa, sb);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] - b.span[i];
    }

    template<typename T0, typename T1>
    void inline mul(AsymmetricSpan<T0, N> const& a, AsymmetricSpan<T1, N> const& b) noexcept
    {
        Span<T0, N> const& sa = a;
        Span<T1, N> const& sb = b;
        Super::mul(sa, sb);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] * b.span[i];
    }

    template<typename T0, typename T1>
    void inline div(AsymmetricSpan<T0, N> const& a, AsymmetricSpan<T1, N> const& b) noexcept
    {
        Span<T0, N> const& sa = a;
        Span<T1, N> const& sb = b;
        Super::div(sa, sb);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] / b.span[i];
    }

    template<typename T0, typename T1, typename T2>
    void inline fma(AsymmetricSpan<T0, N> const& a, AsymmetricSpan<T1, N> const& b,
                    AsymmetricSpan<T2, N> const& c) noexcept
    {
        Span<T0, N> const& sa = a;
        Span<T1, N> const& sb = b;
        Span<T2, N> const& sc = c;
        Super::fma(sa, sb, sc);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] = a.span[i] * b.span[i] + c.span[i];
    }

    template<typename T0>
    auto inline operator+=(AsymmetricSpan<T0, N> const& that) noexcept
    {
        Span<T0, N> const& sthat = that;
        Super::operator+=(sthat);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] += that.span[i];
    }

    template<typename T0>
    auto inline operator-=(AsymmetricSpan<T0, N> const& that) noexcept
    {
        Span<T0, N> const& sthat = that;
        Super::operator-=(sthat);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] -= that[i];
    }

    template<typename T0>
    auto inline operator*=(AsymmetricSpan<T0, N> const& that) noexcept
    {
        Span<T0, N> const& sthat = that;
        Super::operator*=(sthat);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] *= that.span[i];
    }

    template<typename T0>
    auto inline operator/=(AsymmetricSpan<T0, N> const& that) noexcept
    {
        Span<T0, N> const& sthat = that;
        Super::operator/=(sthat);
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            span[i] /= that.span[i];
    }

protected:
    auto modulo_leftover_idx(auto const siz) const { return siz - siz % N; }
    auto modulo_leftover_idx() const { return modulo_leftover_idx(size()); }
};


template<typename T, std::size_t _N = Options::N<std::decay_t<T>>()>
class UnSpan : public AsymmetricSpan<T, _N>
{
    using Super = AsymmetricSpan<T, _N>;
    using R     = std::decay_t<T>;

public:
    using Super::N;
    using AVX_t = mkn::avx::UnType<R, N>;

    UnSpan(auto&&... args)
        : Super{args...}
    {
    }

    template<typename T0>
    void inline operator+=(UnSpan<T0, N> const& that) noexcept
    {
        for (std::size_t i = 0; i < this->batches(); ++i)
        {
            auto const idx = i * N;
            auto v0        = unaligned_load<R, N>(&this->span[idx]);
            v0 += unaligned_load<R, N>(&that.span[idx]);
            unaligned_store(&this->span[idx], v0);
        }
        for (std::size_t i = this->modulo_leftover_idx(); i < this->size(); ++i)
            this->span[i] += that.span[i];
    }

    auto& operator[](std::size_t i) const noexcept { return this->span[i]; }
    auto& operator[](std::size_t i) noexcept { return this->span[i]; }
};



} // namespace mkn::avx

#endif /* _MKN_AVX_SPAN_HPP_ */
