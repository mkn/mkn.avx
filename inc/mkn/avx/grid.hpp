/**
Copyright (c) 2025, Philip Deegan.
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
#ifndef _MKN_AVX_GRID_HPP_
#define _MKN_AVX_GRID_HPP_

#include <cassert>
#include <type_traits>


#include "mkn/kul/math.hpp"
#include "mkn/avx/span.hpp"

namespace mkn::avx
{
template<typename T, std::size_t dimension = 1>
class Grid : public mkn::avx::SpanSet<T>
{
    using Super                    = mkn::avx::SpanSet<T>;
    auto constexpr static is_const = std::is_const_v<T>;

    struct NestedGrid
    {
        using Grid_rt = std::conditional_t<is_const, Grid const&, Grid&>;

        NestedGrid(Grid_rt real, std::size_t const offset)
            : m_real{real}
        {
            for (std::size_t d = 0; d < dimension; ++d)
                m_offset[d] = offset;
            _set_shape();
        }
        NestedGrid(Grid_rt real, std::array<std::size_t, dimension> const offset)
            : m_real{real}
            , m_offset{offset}
        {
            _set_shape();
        }


        Grid_rt m_real;
        std::array<std::size_t, 3> m_offset{0, 0, 0};
        std::array<std::size_t, 3> m_shape{1, 1, 1};
        T* m_data = nullptr;

        template<typename NestedGrid_t>
        void operator*=(NestedGrid_t const& that)
        {
            assert(this->shape() == that.shape());

            for (std::size_t z = m_offset[2]; z < m_shape[2] + m_offset[2]; ++z)
            {
                for (std::size_t y = m_offset[1]; y < m_shape[1] + m_offset[1]; ++y)
                {
                    std::uint32_t const z_offset = z * real().shape()[0] * real().shape()[1];
                    std::uint32_t const y_offset = y * real().shape()[0];
                    std::uint32_t x_pos          = m_offset[0];
                    auto rem_x                   = m_shape[0];

                    while (x_pos <= m_shape[0])
                    {
                        std::uint32_t const place = x_pos + y_offset + z_offset;
                        auto p0                   = &this->real().data()[place];
                        auto const p1             = &that.real().data()[place];
                        bool const are_aligned    = is_aligned(p0) and is_aligned(p1);
                        if (are_aligned)
                        {
                            SpanSet<T> span0{p0, rem_x};
                            SpanSet<const T> span1{p1, rem_x};
                            span0 *= span1;
                            break;
                        }
                        else
                        {
                            *p0 *= *p1;
                        }

                        ++x_pos;
                        --rem_x;
                    }
                }
            }
        };

        template<typename NestedGrid_t>
        void operator+=(NestedGrid_t const& that)
        {
            assert(this->shape() == that.shape());

            for (std::size_t z = m_offset[2]; z < m_shape[2] + m_offset[2]; ++z)
            {
                for (std::size_t y = m_offset[1]; y < m_shape[1] + m_offset[1]; ++y)
                {
                    std::uint32_t const z_offset = z * real().shape()[0] * real().shape()[1];
                    std::uint32_t const y_offset = y * real().shape()[0];
                    std::uint32_t x_pos          = m_offset[0];
                    auto rem_x                   = m_shape[0];

                    while (x_pos <= m_shape[0])
                    {
                        std::uint32_t const place = x_pos + y_offset + z_offset;
                        auto p0                   = &this->real().data()[place];
                        auto const p1             = &that.real().data()[place];
                        bool const are_aligned    = is_aligned(p0) and is_aligned(p1);
                        if (are_aligned)
                        {
                            SpanSet<T> span0{p0, rem_x};
                            SpanSet<T const> span1{p1, rem_x};
                            span0 += span1;
                            break;
                        }
                        else
                        {
                            *p0 += *p1;
                        }

                        ++x_pos;
                        --rem_x;
                    }
                }
            }
        };

        auto& shape() const { return m_shape; }
        auto& real() { return m_real; }
        auto& real() const { return m_real; }

        void _set_shape()
        {
            for (std::size_t d = 0; d < dimension; ++d)
            {
                assert(real().shape()[d] - (2 * m_offset[d]) > 0);
                m_shape[d] = real().shape()[d] - (2 * m_offset[d]);
            }
        }
    };

public:
    Grid(T* data, std::array<std::size_t, dimension> const shape)
        : Super{data, mkn::kul::math::product(shape)}
        , m_data{data}
        , m_shape{shape}
    {
    }

    auto operator>>(std::size_t const offset) { return NestedGrid{*this, offset}; }
    auto operator>>(std::array<std::size_t, dimension> const m_offset)
    {
        return NestedGrid{*this, m_offset};
    }
    auto operator>>(std::size_t const offset) const { return NestedGrid{*this, offset}; }
    auto operator>>(std::array<std::size_t, dimension> const m_offset) const
    {
        return NestedGrid{*this, m_offset};
    }

    auto& shape() const { return m_shape; }
    auto data() { return m_data; }
    auto data() const { return m_data; }

    T* m_data;
    std::array<std::size_t, dimension> m_shape;
};

} // namespace mkn::avx

#endif /* _MKN_AVX_SPAN_HPP_ */
