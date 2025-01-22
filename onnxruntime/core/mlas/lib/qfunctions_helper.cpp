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

// // Copying logic from fxRoundPosInf in cgc_ccl.hpp for custom round
// template <uint8_t aFracBits>
// inline int32_t customRound(const int32_t a) {
//     const int32_t zp5 = 1 << (aFracBits - 1);
//     return (a + zp5) >> aFracBits;
// }

// template int32_t customRound<2>(const int32_t);

// Copying logic from data_to_qfp in tvm/python/tvm/target/epu_fx_util.py
// For purposes of calculating number of frac bits needed to represent scale in quantize ops

// Function to derive fractional bits
int deriveFractionalBits(float scalar, int qfpSize) {
    int valueBits = qfpSize - 1;

    float intPart;
    ::modff(scalar, &intPart); // Returns the frac part which we dont care about, int part gets stored in pointer
    intPart = std::abs(intPart);

    int intBits = (intPart == 0) ? 0 : static_cast<int>(std::log2f(intPart)) + 1;
    int fracBits = valueBits - intBits;

    assert(fracBits >= 0 && "Scalar cannot be represented in qfp format.");

    return fracBits;
}

// Function to convert scalar to qfp
int scalarToQfp(float value, int fracBits) {
    float frac, integer;
    frac = ::modff(value, &integer);

    integer = static_cast<int>(std::abs(integer)) << fracBits;
    frac = std::roundf(std::abs(frac) * (1 << fracBits));

    int qfp = static_cast<int>(integer + frac);
    if (value < 0) {
        qfp *= -1;
    }

    return qfp;
}

// // Function to convert data to qfp
// // In quantize.cpp, we want to pass in scale, which is a pointer to a float
// // Here the function accepts a vector (because if it were a pointer we would have to pass in size also)
// // When using this we can just turn whatever into std::vector<T> before passing in
// template <typename T>
// std::pair<std::vector<int>, int> dataToQfp(
//     const std::vector<T>& data, int fracBits = -1, int qfpSize = 32, bool scalarAsFloat = true
// ) {
//     auto deriveFractionalBits = [qfpSize](float scalar) {
//         int valueBits = qfpSize - 1;

//         float intPart;
//         ::modff(scalar, &intPart);
//         intPart = std::abs(intPart);

//         int intBits = (intPart == 0) ? 0 : static_cast<int>(std::log2f(intPart)) + 1;
//         int fracBits = valueBits - intBits;

//         assert(fracBits >= 0 && "Scalar cannot be represented in qfp format.");

//         return fracBits;
//     };

//     auto scalarToQfp = [](float value, int fracBits) {
//         float frac, integer;
//         frac = ::modff(value, &integer);

//         integer = static_cast<int>(std::abs(integer)) << fracBits;
//         frac = std::roundf(std::abs(frac) * (1 << fracBits));

//         int qfp = static_cast<int>(integer + frac);
//         if (value < 0) {
//             qfp *= -1;
//         }

//         return qfp;
//     };

//     std::vector<int> qfp;
//     if (data.size() != 1) {
//         if (fracBits == -1) {
//             fracBits = deriveFractionalBits(*std::max_element(data.begin(), data.end(), [](T a, T b) { return std::abs(a) < std::abs(b); }));
//         }
//         qfp.reserve(data.size());
//         std::transform(data.begin(), data.end(), std::back_inserter(qfp), [fracBits, &scalarToQfp](T value) {
//             return scalarToQfp(value, fracBits);
//         });
//     } else { // data is not really a vector, but we considered everything a vector in our declaration
//         if (fracBits == -1) {
//             fracBits = deriveFractionalBits(data[0]);
//         }
//         if (scalarAsFloat) {
//             // **In the case where the value is an immediate return
//             // it as float to be consumed directly by the codegen** from epu_fx_util.py
//             qfp.push_back(static_cast<int>(data[0]));
//         } else {
//             // **Case when converting to integer constant in Relay to enable
//             // post quantized CPU inference** from epu_fx_util.py
//             qfp.push_back(scalarToQfp(data[0], fracBits));
//         }
//     }

//     return std::make_pair(qfp, fracBits);
// }
