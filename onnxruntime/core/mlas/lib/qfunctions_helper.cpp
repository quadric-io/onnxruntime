#include "mlasi.h"
#include <iostream>

#include <cmath>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

// Copying logic from data_to_qfp in tvm/python/tvm/target/epu_fx_util.py
// For purposes of calculating number of frac bits needed to represent scale in quantize ops

// Function to derive fractional bits
int deriveFractionalBits(double scalar, int qfpSize) {
    int valueBits = qfpSize - 1;

    double intPart;
    ::modf(scalar, &intPart); // Returns the frac part which we dont care about, int part gets stored in pointer
    intPart = std::abs(intPart);

    int intBits = (intPart == 0) ? 0 : static_cast<int>(std::log2f(intPart)) + 1;
    int fracBits = valueBits - intBits;

    assert(fracBits >= 0 && "Scalar cannot be represented in qfp format.");

    return fracBits;
}

// Function to convert scalar to qfp
int scalarToQfp(double value, int fracBits) {
    double frac, integer;
    frac = ::modf(value, &integer);

    integer = static_cast<int>(std::abs(integer)) << fracBits;
    frac = std::roundf(std::abs(frac) * (1 << fracBits));

    int qfp = static_cast<int>(integer + frac);
    if (value < 0) {
        qfp *= -1;
    }

    return qfp;
}
