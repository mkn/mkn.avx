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
#ifndef _MKN_AVX_VECTOR_HPP_
#define _MKN_AVX_VECTOR_HPP_

#include "mkn/avx/def.hpp"
#include "mkn/avx/span.hpp"

#include <vector>
#include <optional>

namespace mkn::avx
{
template<typename T, typename Allocator_ = typename std::vector<T>::allocator_type>
struct _V_
{
    using Allocator = Allocator_;
    using vec_t     = std::vector<T, Allocator>;

    _V_(std::size_t s, T val = 0)
        : vec(s, val)
    {
    }

    vec_t vec;
};

template<typename T, typename Allocator = typename std::vector<T>::allocator_type>
class Vector : public _V_<T, Allocator>, public Span<T>
{
    using This = Vector<T, Allocator>;

public:
    auto constexpr static N = Options::N<T>();
    using Vec               = _V_<T, Allocator>;
    using Vec::vec;

    Vector(std::size_t s = 0, T val = 0)
        : Vec(s, val)
        , Span<T>{Vec::vec.data(), Vec::vec.size()}
    {
    }


    auto operator+(This const& that) noexcept
    {
        using AVX_t = typename Span<T>::AVX_t;

        assert(this->size() >= that.size());

        Vector<T> r(this->size());

        auto& v0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&(*this)());
        auto& v1 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&that());
        auto& r0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&r());

        for (std::size_t i = 0; i < this->size() / N; ++i)
            r0[i] = v0[i] + v1[i];

        for (std::size_t i = this->size() - this->size() % N; i < this->size(); ++i)
            r[i] = (*this)[i] + that[i];

        return r;
    }

    auto operator*(This const& that) noexcept
    {
        using AVX_t = typename Span<T>::AVX_t;

        assert(this->size() >= that.size());

        Vector<T> r(this->size());

        auto& v0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&(*this)());
        auto& v1 = *reinterpret_cast<mkn::kul::Span<AVX_t> const*>(&that());
        auto& r0 = *reinterpret_cast<mkn::kul::Span<AVX_t>*>(&r());

        for (std::size_t i = 0; i < this->size() / N; ++i)
            r0[i] = v0[i] * v1[i];

        for (std::size_t i = this->size() - this->size() % N; i < this->size(); ++i)
            r[i] = (*this)[i] * that[i];

        return r;
    }
};

} /* namespace mkn::avx */

#endif /* _MKN_AVX_VECTOR_HPP_ */
