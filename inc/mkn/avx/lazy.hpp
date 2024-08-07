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

#include "mkn/kul/io.hpp"
#include "mkn/kul/log.hpp"
#include "mkn/kul/alloc.hpp"
#include "mkn/avx/span.hpp"

#include <new>
#include <tuple>
#include <vector>

namespace mkn::avx
{

template<typename T, typename Small = std::uint8_t /* use std::uint16_t for printing */>
struct LazyOp
{
    LazyOp(T* _a, T const* _b, std::size_t const& _op)
        : a{_a}
        , b{_b}
        , op{_op}
    {
    }


    T* a;
    T const* b;
    std::size_t op;
    Small t      = 0;
    LazyOp* prev = nullptr;
    LazyOp* next = nullptr;
};

template<typename T>
struct LazyVal
{
    static thread_local inline std::size_t alive = 0;
    using value_type                             = T;
    using This                                   = LazyVal<T>;

    LazyVal(T& t)
        : v{&t}
    {
    }
    ~LazyVal() {}

    LazyVal(LazyVal const& that) = default;
    LazyVal(LazyVal&& that)      = default;


    auto operator+(This const& that) const
    {
        operands.emplace_back(v, that.v, 0);
        return *this;
    }
    auto operator-(This const& that) const
    {
        operands.emplace_back(v, that.v, 1);
        return *this;
    }

    auto operator*(This const& that) const
    {
        operands.emplace_back(v, that.v, 2);
        return *this;
    }
    auto operator/(This const& that) const
    {
        operands.emplace_back(v, that.v, 3);
        return *this;
    }

    auto& operator()() { return *v; }
    auto& operator()() const { return *v; }


    bool muldiv(std::size_t const& i) const { return operands[i].op == 2 or operands[i].op == 3; }

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

    LazyEvaluator(LazyVal_t& _t)
        : t{_t}
    {
    }

    ~LazyEvaluator() { clear(); }

    void clear()
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

    void compile()
    {
        tmps.resize(t.operands.size() - 1);
        for (size_t i = t.operands.size(); i-- > 0;)
        {
            auto& op = t.operands[i];

            for (size_t j = i; j-- > 0;)
            {
                auto& jop = t.operands[j];
                if (jop.next)
                    continue;
                if (jop.a == op.b)
                {
                    op.prev  = &jop;
                    jop.next = &op;

                    break;
                }
            }
        }
    }

    auto operator()(T* const ret, bool fill = false)
    {
        compile();
        auto const& size = t.operands[0].a->size();

        if (fill)
            std::copy(t.operands[0].a->data(), t.operands[0].a->data() + size, ret);

        std::size_t tmp = 0;
        auto do_avx     = [&](auto const& o, auto const& i) {
            std::size_t const off = i * N;
            assert(off < size);

            auto& op           = t.operands[o];
            bool const use_tmp = op.a != t.v;
            Span_ct const a{op.a->data() + off, N};
            Span_ct const b{op.b->data() + off, N};
            Span_t r{ret + off, N};

            if (op.prev)
            {
                if (use_tmp)
                {
                    Span_ct const pspan{tmps[op.prev->t].data(), N};
                    Span_t tspan{tmps[tmp].data(), N};
                    fns[op.op](tspan, a, pspan);

                    op.t = tmp++;
                }
                else
                {
                    if (op.prev->a == t.v)
                    {
                        fns[op.op](r, r, b);
                    }
                    else
                    {
                        Span_t tspan{tmps[op.prev->t].data(), N};
                        fns[op.op](r, r, tspan);
                    }
                }
            }
            else
            {
                if (use_tmp)
                {
                    Span_t tspan{tmps[tmp].data(), N};
                    fns[op.op](tspan, a, b);

                    op.t = tmp++;
                }
                else
                {
                    fns[op.op](r, r, b);
                }
            }
        };

        auto const cl_size = std::hardware_destructive_interference_size;

        std::size_t const batch = cl_size / sizeof(T) / N;

        std::size_t i = 0;
        for (; i < size / N; i += batch, tmp = 0)
            for (std::size_t o = 0; o < t.operands.size(); ++o)
                for (std::size_t j = 0; j < batch; ++j)
                    do_avx(o, i + j);

        if (size % N != 0)
            for (; i < size / N + 1; i += batch, tmp = 0)
                for (std::size_t o = 0; o < t.operands.size(); ++o)
                    do_avx(o, i);
    }


    void write_compilable(std::string const& fileout)
    {
        kul::io::Writer w{fileout};
        std::string const padding = "        ";
        std::string const header  = R"(
#include "mkn/kul/alloc.hpp"
#include "mkn/avx/span.hpp"

#include <array>
#include <vector>
#include <cstdint>

template<typename E>
using AVXVec = std::vector<E, mkn::kul::AlignedAllocator<E, 32>>;
)";

        std::string const funcheader = R"(

template<typename LazyVal_t, typename T>
void exec(LazyVal_t const& t, T* const ret){
    using Span_t            = mkn::avx::Span<T>;
    using Span_ct           = mkn::avx::Span<T const>;
    auto constexpr static N = mkn::avx::Options::N<T>();
    static AVXVec<std::array<T, N>> tmps(n_tmps);
    std::fill_n(tmps[0].data(), N * tmps.size(), 0);
    auto const& ops = t.operands;
    auto const& size = t.operands[0].a->size();
    for (std::size_t i = 0; i < size / N; ++i)
    {
        std::size_t const off = i * N;
        Span_t r{ret + off, N};
)";

        std::stringstream body;
        body << "\n";
        std::size_t tmp = 0;
        for (std::size_t o = 0; o < t.operands.size(); ++o)
        {
            auto& op            = t.operands[o];
            bool const use_tmp  = op.a != t.v;
            std::string const a = "Span_ct{ops[" + std::to_string(o) + "].a->data() + off, N}";
            std::string const b = "Span_ct{ops[" + std::to_string(o) + "].b->data() + off, N}";

            if (op.prev)
            {
                if (use_tmp)
                {
                    body << padding << "Span_t{tmps[" << tmp << "].data(), N}." << fn_strs[op.op]
                         << "(" << a << ", Span_ct{tmps[" << op.prev->t << "].data(), N});"
                         << mkn::kul::os::EOL();

                    op.t = tmp++;
                }
                else
                {
                    if (op.prev->a == t.v)
                    {
                        body << padding << "r." << fn_strs[op.op] << "(r, " << b << ");"
                             << mkn::kul::os::EOL();
                    }
                    else
                    {
                        body << padding << "r." << fn_strs[op.op] << "(r, Span_ct{tmps["
                             << op.prev->t << "].data(), N});" << mkn::kul::os::EOL();
                    }
                }
            }
            else
            {
                if (use_tmp)
                {
                    body << padding << "Span_t{tmps[" << tmp << "].data(), N}." << fn_strs[op.op]
                         << "(" << a << ", " << b << ");" << mkn::kul::os::EOL();
                    op.t = tmp++;
                }
                else
                {
                    body << padding << "r." << fn_strs[op.op] << "(r, " << b << ");"
                         << mkn::kul::os::EOL();
                }
            }
        }

        w << header;
        w << "\nconstexpr static std::size_t n_tmps = " << (tmp) << ";";
        w << funcheader;
        w << body.str();
        w << R"(    }
}

)";
    }

    LazyVal_t& t;
    std::vector<std::function<void(Span_t&, Span_ct const&, Span_ct const&)>> fns{__add__, __sub__,
                                                                                  __mul__, __div__};
    std::vector<std::string> fn_strs{"add", "sub", "mul", "div"};

    template<typename E>
    using AVXVec = std::vector<E, mkn::kul::AlignedAllocator<E, 32>>;
    static inline thread_local AVXVec<std::array<T, N>> tmps{};
};

template<typename T>
auto eval(LazyVal<T>& v, bool in_place = false)
{
    auto ret = v();
    LazyEvaluator<LazyVal<T>>{v}(ret.data());
    return ret;
}
template<typename T>
auto eval(LazyVal<T>&& v, bool in_place = false)
{
    auto ret = v();
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
