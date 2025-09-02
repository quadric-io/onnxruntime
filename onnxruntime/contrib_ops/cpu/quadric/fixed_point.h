// This is a common header file for fixed point computations. These are specifically for small functions that could be inlined.

#include <cmath>
#include <algorithm>
#include "core/common/common.h"

namespace chimera {
// Fixed-point multiplication with provided shift
inline std::int32_t fixedPointMultiply(std::int32_t a, std::int32_t b, std::int8_t shift) {
  std::int64_t product = (std::int64_t)a * (std::int64_t)b;
  return (shift > 0) ? (product >> shift) : (product << -shift);
}

// fixed point division
inline std::int32_t fixedPointDiv(std::int32_t op1, std::int32_t op2, std::int8_t preShift) {
  return std::int32_t((std::int64_t(op1) << preShift) / std::int64_t(op2));
}

inline std::int32_t log2Ceil(std::int32_t n) {
  std::int32_t isPowerOf2 = (n & (n - 1)) == 0;
  return log2(n) + !isPowerOf2;
}

// sqare root function in fixed-point.
template <typename T>
inline std::int32_t sqrt(T qInput, std::int8_t in_fbits, std::int32_t out_fbits) {
  std::int32_t ResVal = 0;

  std::int32_t bitShiftAmount = (out_fbits - in_fbits) + out_fbits;
  std::int64_t InputIntermediateVal = std::int64_t(qInput) << bitShiftAmount;
  std::int8_t startBit = (((sizeof(T) * 8 - bitShiftAmount) >> 1) + bitShiftAmount - 1);
  for (std::int8_t i = startBit; i >= 0; --i) {
    std::int64_t CandidateVal = (1 << i) + ResVal;
    std::int64_t IntermediateVal = CandidateVal * CandidateVal;
    if (IntermediateVal <= InputIntermediateVal) {
      ResVal = CandidateVal;
    }
  }
  return ResVal;
}
}  // namespace chimera
