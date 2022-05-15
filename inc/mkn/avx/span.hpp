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

#include "mkn/kul/log.hpp"
#include "mkn/kul/span.hpp"

#include "mkn/avx/def.hpp"
#include "mkn/avx/types.hpp"


namespace mkn::avx
{
template<typename T, std::uint16_t _N = Options::N<std::decay_t<T>>()>
class Span : public mkn::kul::Span<T>
{
    using R     = std::decay_t<T>;
    using Super = mkn::kul::Span<T>;

public:
    using Super::size;
    using Super::data;

    using value_type        = T;
    auto constexpr static N = _N;

    using AVX_t = mkn::avx::Type<T, N>;

    Span(T* d, std::size_t s) noexcept
        : Super{d, s}
    {
    }

    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    Span(C const& v) noexcept
        : Super{v}
    {
    }
    template<typename C, std::enable_if_t<kul::is_span_like_v<C>, bool> = 0>
    Span(C& v) noexcept
        : Super{v}
    {
    }


    template<typename T0, typename T1>
    void add(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        auto& v0 = caster(*this);
        auto& v1 = caster(a);
        auto& v2 = caster(b);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] + v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] = a[i] + b[i];
    }

    template<typename T0, typename T1>
    void mul(Span<T0, N> const& a, Span<T1, N> const& b) noexcept
    {
        static_assert(sizeof(mkn::kul::Span<AVX_t>) == sizeof(mkn::kul::Span<T>));

        auto& v0 = caster(*this);
        auto& v1 = caster(a);
        auto& v2 = caster(b);

        assert(v0.size() == v1.size() and v1.size() == v2.size());
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] = v1[i] * v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] = a[i] * b[i];
    }


    template<typename T0>
    void operator+=(Span<T0, N> const& that) noexcept
    {
        assert(this->size() <= that.size());

        auto& v0 = caster(*this);
        auto& v1 = caster(that);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] += v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] += that[i];
    }

    template<typename T0>
    void operator+=(std::array<T0, N> const& arr) noexcept
    {
        // KLOG(INF) << size() << " " << N;

        assert(this->size() % N == 0);

        Span<T0 const, N> that{arr};
        auto& v0 = caster(*this);
        auto& v1 = caster(that);

        // KLOG(INF) << (*this)[0] << " " << arr[0];
        for (std::size_t i = 0; i < size() / N; ++i)
        {
            // KLOG(INF) << i;
            v0[i] += v1[0]; // v1 only has one set of elements
        }
    }


    void operator+=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);

        Span<T const, N> that{scratch};
        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] += v1[0]; // v1 only has one set of elements

        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] += val;
    }


    template<typename T0>
    void operator-=(Span<T0, N> const& that) noexcept
    {
        assert(this->size() <= that.size());

        auto& v0 = caster(*this);
        auto& v1 = caster(that);

        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] -= v1[i];
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] -= that[i];
    }

    template<typename T0>
    void operator-=(std::array<T0, N> const& arr) noexcept
    {
        // KLOG(INF) << size() << " " << N;

        assert(this->size() % N == 0);

        Span<T0 const, N> that{arr};
        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] -= v1[0]; // v1 only has one set of elements
    }


    void operator-=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);

        Span<T const, N> that{scratch};
        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] -= v1[0]; // v1 only has one set of elements

        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] -= val;
    }


    template<typename T0>
    void operator*=(Span<T0, N> const& that) noexcept
    {
        // KLOG(INF) << this->size() << " " << N;
        assert(this->size() <= that.size());

        static_assert(sizeof(mkn::kul::Span<AVX_t>) == sizeof(mkn::kul::Span<T>));

        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
        {
            // KLOG(INF) << i << " " << (i * N);
            v0[i] *= v1[i];
        }
        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] *= that[i];
    }

    template<typename T0>
    void operator*=(std::array<T0, N> const& arr) noexcept
    {
        Span<T0 const, N> that{arr};
        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] *= v1[0]; // v1 only has one set of elements

        if (this->size() % N != 0)
            for (std::size_t i = modulo_leftover_idx(), j = 0; i < size(); ++i, ++j)
                (*this)[i] *= arr[j];
    }

    void operator*=(T const& val) noexcept
    {
        std::fill(scratch.begin(), scratch.end(), val);

        Span<T const, N> that{scratch};
        auto& v0 = caster(*this);
        auto& v1 = caster(that);
        for (std::size_t i = 0; i < size() / N; ++i)
            v0[i] *= v1[0]; // v1 only has one set of elements

        for (std::size_t i = modulo_leftover_idx(); i < size(); ++i)
            (*this)[i] *= val;
    }



    template<typename T0>
    bool operator==(Span<T0, N> const& that) const noexcept
    { // TODO vectorize
        for (std::size_t i = 0; i < size(); ++i)
            if ((*this)[i] != that[i])
                return false;
        return true;
    }

    bool operator==(T const t) const noexcept
    { // TODO vectorize
        for (std::size_t i = 0; i < size(); ++i)
            if ((*this)[i] != t)
                return false;
        return true;
    }


protected:
    auto modulo_leftover_idx() { return size() - size() % N; }

private:
    std::array<T, N> scratch{};

    template<typename T0, typename = std::enable_if_t<std::is_same_v<R, std::decay_t<T0>>>>
    static auto& caster(Span<T0, N>& that) noexcept
    {
        return *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&that);
    }

    template<typename T0, typename = std::enable_if_t<std::is_same_v<R, std::decay_t<T0>>>>
    static auto& caster(Span<T0, N> const& that) noexcept
    {
        return *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&that);
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
