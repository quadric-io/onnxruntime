#include <cstdint> // For uint8_t, int32_t, etc.
#include <vector>
#include <utility>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

// Copying logic from fxRoundPosInf in cgc_ccl.hpp for custom round
template <uint8_t aFracBits>
inline int32_t customRound(const int32_t a) {
    const int32_t zp5 = 1 << (aFracBits - 1);
    return (a + zp5) >> aFracBits;
}

// Function to derive fractional bits
int deriveFractionalBits(double scalar, int qfpSize);

// Function to convert scalar to qfp
int scalarToQfp(double value, int fracBits);

// Function to convert data to qfp
template <typename T>
std::pair<std::vector<int>, int> dataToQfp(
    const std::vector<T>& data, int fracBits = -1, int qfpSize = 32, bool scalarAsFloat = true
) {
    auto deriveFractionalBits = [qfpSize](double scalar) {
        int valueBits = qfpSize - 1;

        double intPart;
        ::modf(scalar, &intPart);
        intPart = std::abs(intPart);

        int intBits = (intPart == 0) ? 0 : static_cast<int>(std::log2f(intPart)) + 1;
        int fracBits = valueBits - intBits;

        assert(fracBits >= 0 && "Scalar cannot be represented in qfp format.");

        return fracBits;
    };

    auto scalarToQfp = [](double value, int fracBits) {
        double frac, integer;
        frac = ::modf(value, &integer);

        integer = static_cast<int>(std::abs(integer)) << fracBits;
        frac = std::roundf(std::abs(frac) * (1 << fracBits));

        int qfp = static_cast<int>(integer + frac);
        if (value < 0) {
            qfp *= -1;
        }

        return qfp;
    };

    std::vector<int> qfp;
    if (data.size() != 1) {
        if (fracBits == -1) {
            fracBits = deriveFractionalBits(*std::max_element(data.begin(), data.end(), [](T a, T b) { return std::abs(a) < std::abs(b); }));
        }
        qfp.reserve(data.size());
        std::transform(data.begin(), data.end(), std::back_inserter(qfp), [fracBits, &scalarToQfp](T value) {
            return scalarToQfp(value, fracBits);
        });
    } else {
        if (fracBits == -1) {
            fracBits = deriveFractionalBits(data[0]);
        }
        if (scalarAsFloat) {
            qfp.push_back(static_cast<int>(data[0]));
        } else {
            qfp.push_back(scalarToQfp(data[0], fracBits));
        }
    }

    return std::make_pair(qfp, fracBits);
}
