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
#ifndef _MKN_AVX_GRID_HPP_
#define _MKN_AVX_GRID_HPP_

#include <cassert>

#include "mkn/avx/span.hpp"

namespace mkn::avx
{
template<typename T, std::size_t dimension = 1>
class Grid : public mkn::avx::Span<T>
{
    using Super = mkn::avx::Span<T>;

    struct NestedGrid
    {
        NestedGrid(Grid& real, std::size_t offset)
            : m_real{real}
        {
            for (std::size_t d = 0; d < dimension; ++d)
                m_offset[d] = offset;
            _set_shape();
        }
        NestedGrid(Grid& real, std::array<std::size_t, dimension> offset)
            : m_real{real}
            , m_offset{offset}
        {
            _set_shape();
        }


        Grid& m_real;
        std::array<std::size_t, 3> m_offset{0, 0, 0};
        std::array<std::size_t, 3> m_shape{1, 1, 1};
        T* m_data = nullptr;

        void operator*=(NestedGrid const& that)
        {
            assert(this->shape() == that.shape());

            for (std::size_t z = m_offset[2]; z < m_shape[2] + m_offset[2]; ++z)
            {
                for (std::size_t y = m_offset[1]; y < m_shape[1] + m_offset[1]; ++y)
                {
                    std::uint32_t z_offset = z * real().shape()[0] * real().shape()[1];
                    std::uint32_t y_offset = y * real().shape()[0];
                    std::uint32_t place    = m_offset[0] + y_offset + z_offset;

                    auto p0 = &this->real().data()[place];
                    auto p1 = &that.real().data()[place];

                    Span<T> span0{p0, m_shape[0]};
                    Span<T> span1{p1, m_shape[0]};

                    span0 *= span1;
                }
            }
        };


        void operator+=(NestedGrid const& that)
        {
            assert(this->shape() == that.shape());

            for (std::size_t z = m_offset[2]; z < m_shape[2] + m_offset[2]; ++z)
            {
                for (std::size_t y = m_offset[1]; y < m_shape[1] + m_offset[1]; ++y)
                {
                    KLOG(INF) << z << " " << y;

                    std::uint32_t z_offset = z * real().shape()[0] * real().shape()[1];
                    std::uint32_t y_offset = y * real().shape()[0];
                    std::uint32_t place    = m_offset[0] + y_offset + z_offset;

                    auto p0 = &this->real().data()[place];
                    auto p1 = &that.real().data()[place];

                    Span<T> span0{p0, m_shape[0]};
                    Span<T> span1{p1, m_shape[0]};

                    KLOG(INF) << place << " " << m_shape[0];
                    span0 += span1;
                }
            }
        };

        auto& shape() const { return m_shape; }
        auto& real() { return m_real; }
        auto& real() const { return m_real; }

        void _set_shape()
        {
            for (std::size_t d = 0; d < dimension; ++d)
                m_shape[d] = real().shape()[d] - (2 * m_offset[d]);
        }
    };

public:
    Grid(T* data, std::array<std::size_t, dimension> shape)
        : Super{data, mkn::kul::math::product(shape)}
        , m_data{data}
        , m_shape{shape}
    {
    }

    NestedGrid operator>>(std::size_t const offset) { return NestedGrid{*this, offset}; }
    NestedGrid operator>>(std::array<std::size_t, dimension> m_offset)
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
