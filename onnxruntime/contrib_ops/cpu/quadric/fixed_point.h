// This is a common header file for fixed point computations.

#include "core/common/common.h"
#include <algorithm>

// Fixed-point multiplication with provided shift
std::int32_t fixedPointMultiply(std::int32_t a, std::int32_t b, std::int8_t shift) {
  std::int64_t product = (std::int64_t)a * (std::int64_t)b;
  return (shift > 0) ? (product >> shift) : (product << -shift);
}
