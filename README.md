# mkn.avx

**C++20 AVX wrappers**

Header-only, AVX1 / AVX2 / AVX512 are activated from the compiler's target flags.

## Example

```cpp
#include "mkn/avx.hpp"
#include "mkn/avx/array.hpp"

int main() {
  using Array = mkn::avx::Array<double, mkn::avx::Options::N<double>()>;
  Array a{1}, b{2};
  a += b;  // a == 3
  a *= b;  // a == 6
  return !(a == 6);
}
```

`-O3` or higher is required for the intrinsics to be used, e.g. `-O3 -march=native`.

See [test/test_avx.cpp](test/test_avx.cpp) for more.

## Compile switches

| define | type | default | description |
|---|---|---|---|
| `MKN_AVX_1_ACTIVE` | bool | true if `__AVX__` | activate AVX1 |
| `MKN_AVX_2_ACTIVE` | bool | true if `__AVX2__` | activate AVX2 (256 bit) |
| `MKN_AVX_512_ACTIVE` | bool | true if `__AVX512F__` | activate AVX512 |
| `MKN_AVX_ALIGN_AS` | int | 64 avx512, 32 avx2, 16 avx1, else 8 | alignment override |

## Usage

In a project `mkn.yaml`:

```yaml
dep: mkn.avx
```

Build and run the tests and benchmark:

```bash
mkn clean build test -gdtKOp test,test_lazy,gbench -l -pthread dbg
```

## Dependencies

[mkn.kul](https://github.com/mkn/mkn.kul)
