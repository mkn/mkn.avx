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
#ifndef _MKN_AVX_LAZY_HPP_
#define _MKN_AVX_LAZY_HPP_

#include "mkn/kul/log.hpp"
#include "mkn/kul/alloc.hpp"
#include "mkn/kul/threads.hpp"
#include "mkn/avx/span.hpp"

#include <tuple>
#include <vector>

namespace mkn::avx
{

template<typename T>
struct LazyOp
{
    auto constexpr static N = mkn::avx::Options::N<T>();

    T* a;
    T const* b;
    std::size_t op;
    std::uint16_t id;

    LazyOp* prev = nullptr;
    bool marked  = false;
};

template<typename T>
struct LazyVal
{
    using value_type = T;
    using This       = LazyVal<T>;

    LazyVal(T& t)
        : v{&t}
    {
    }
    LazyVal(LazyVal const& that)
        : v{that.v}
    {
    }
    LazyVal(LazyVal&&) = default;

    auto operator+(This const& that) const
    {
        operands.emplace_back(v, that.v, 0, operands.size());
        return *this;
    }
    auto operator-(This const& that) const
    {
        operands.emplace_back(v, that.v, 1, operands.size());
        return *this;
    }

    auto operator*(This const& that) const
    {
        operands.emplace_back(v, that.v, 2, operands.size());
        return *this;
    }
    auto operator/(This const& that) const
    {
        operands.emplace_back(v, that.v, 3, operands.size());
        return *this;
    }

    auto& operator()() { return *v; }
    auto& operator()() const { return *v; }

    T* v;
    static inline thread_local std::vector<LazyOp<T>> operands;
};

template<typename LazyVal_t>
struct LazyEvaluator
{
    using Vec_t             = typename LazyVal_t::value_type;
    using T                 = typename Vec_t::value_type;
    using Span_t            = mkn::avx::Span<T>;
    using Span_ct           = mkn::avx::Span<T const>;
    auto constexpr static N = mkn::avx::Options::N<T>(); // max vector size

    ~LazyEvaluator()
    {
        tmps.clear();
        LazyVal_t::operands.clear();
    }

    static constexpr inline auto __add__
        = [](Span_t& r, Span_ct const& a, Span_ct const& b) mutable { r.add(a, b); };
    static constexpr inline auto __sub__
        = [](Span_t& r, Span_ct const& a, Span_ct const& b) mutable { r.sub(a, b); };
    static constexpr inline auto __mul__
        = [](Span_t& r, Span_ct const& a, Span_ct const& b) mutable { r.mul(a, b); };
    static constexpr inline auto __div__
        = [](Span_t& r, Span_ct const& a, Span_ct const& b) mutable { r.div(a, b); };
    static constexpr inline auto __fma__ = [](Span_t& r, Span_ct const& a, Span_ct const& b,
                                              Span_ct const& c) mutable { r.fma(a, b, c); };

    void compile()
    {
        if (not fns.size())
        {
            fns.emplace_back(__add__);
            fns.emplace_back(__sub__);
            fns.emplace_back(__mul__);
            fns.emplace_back(__div__);
            // fns.emplace_back(__fma__);
        }
        tmps.resize(t.operands.size());
        for (size_t i = t.operands.size(); i-- > 0;)
        {
            auto& op = t.operands[i];
            for (size_t j = i; j-- > 0;)
            {
                auto& jop = t.operands[j];
                if (jop.marked)
                    continue;
                if (jop.a == op.b)
                {
                    op.prev    = &jop;
                    jop.marked = 1;
                    break;
                }
            }
            if (op.prev)
                continue;
        }
    }

    auto operator()(T* const ret, bool fill = false)
    {
        compile();
        auto const& size = t.operands[0].a->size();

        if (fill)
            std::copy(t.operands[0].a->data(), t.operands[0].a->data() + size, ret);

        for (std::size_t i = 0; i < size / N; ++i)
        {
            std::size_t const off = i * N;
            assert(off < size);

            for (std::size_t o = 0; o < t.operands.size(); ++o)
            {
                auto const& op = t.operands[o];
                if (op.op > 10)
                    continue;
                bool const use_tmp = op.a != t.v;
                Span_ct const a{op.a->data() + off, N};
                Span_ct const b{op.b->data() + off, N};
                Span_t r{ret + off, N};

                //   KLOG(INF) << a[0] << " " << b[0] << " " << r[0] << " " << use_tmp << " "
                //           << bool{op.prev};

                if (use_tmp)
                {
                    if (op.prev)
                    {
                        Span_ct const pspan{tmps[op.prev->id].data(), N};
                        Span_t tspan{tmps[o].data(), N};
                        fns[op.op](tspan, a, pspan);
                    }
                    else
                    {
                        Span_t tspan{tmps[o].data(), N};

                        fns[op.op](tspan, a, b);
                    }
                }
                else
                {
                    if (op.prev)
                    {
                        Span_ct const tspan{tmps[op.prev->id].data(), N};
                        fns[op.op](r, r, tspan);
                    }
                    else
                    {
                        fns[op.op](r, r, b);
                    }
                }
            }
        }
    }


    LazyVal_t& t;
    std::vector<std::function<void(Span_t&, Span_ct const&, Span_ct const&)>> fns{};

    template<typename E>
    using AVXVec = std::vector<E, mkn::kul::AlignedAllocator<E, 32>>;
    static inline thread_local AVXVec<std::array<T, N>> tmps{};
};

template<typename T>
auto eval(LazyVal<T>& v, bool in_place = false)
{
    auto ret = v(); // copy
    LazyEvaluator<LazyVal<T>>{v}(ret.data());
    return ret;
}
template<typename T>
auto eval(LazyVal<T>&& v, bool in_place = false)
{
    auto ret = v(); // copy
    LazyEvaluator<LazyVal<T>>{v}(ret.data());
    return ret;
}

template<typename... T>
auto lazy(T&... v)
{
    return std::make_tuple(LazyVal<T>{v}...);
}

} // namespace mkn::avx

#endif /* _MKN_AVX_LAZY_HPP_ */
