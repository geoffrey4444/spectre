// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm {

/*!
 * \brief Transform scalar-Ylm coefficients into tensor-Ylm coefficients.
 *
 * The `scalar_ylm_coefficients` are stored in Spherepack ordering, with
 * `number_of_offsets` contiguous values per spectral coefficient.
 * The tensor structure of `TensorType` determines the tensor-Ylm sectors that
 * are produced.
 */
template <typename TensorType>
void scalar_to_tensor_ylm_coefficients(
    gsl::not_null<TensorType*> result,
    const TensorType& scalar_ylm_coefficients, size_t l_max,
    size_t number_of_offsets = 1,
    CoefficientNormalization coefficient_normalization =
        CoefficientNormalization::Spherepack);

/*!
 * \brief Return tensor-Ylm coefficients transformed from scalar-Ylm
 * coefficients.
 *
 * See the `gsl::not_null` overload for coefficient layout and option
 * semantics.
 */
template <typename TensorType>
TensorType scalar_to_tensor_ylm_coefficients(
    const TensorType& scalar_ylm_coefficients, size_t l_max,
    size_t number_of_offsets = 1,
    CoefficientNormalization coefficient_normalization =
        CoefficientNormalization::Spherepack);

/*!
 * \brief Transform tensor-Ylm coefficients into scalar-Ylm coefficients.
 *
 * The `tensor_ylm_coefficients` and the output `result` use Spherepack
 * coefficient ordering, with `number_of_offsets` contiguous values per
 * spectral coefficient.
 */
template <typename TensorType>
void tensor_to_scalar_ylm_coefficients(
    gsl::not_null<TensorType*> result,
    const TensorType& tensor_ylm_coefficients, size_t l_max,
    size_t number_of_offsets = 1,
    CoefficientNormalization coefficient_normalization =
        CoefficientNormalization::Spherepack);

/*!
 * \brief Return scalar-Ylm coefficients transformed from tensor-Ylm
 * coefficients.
 *
 * See the `gsl::not_null` overload for coefficient layout and option
 * semantics.
 */
template <typename TensorType>
TensorType tensor_to_scalar_ylm_coefficients(
    const TensorType& tensor_ylm_coefficients, size_t l_max,
    size_t number_of_offsets = 1,
    CoefficientNormalization coefficient_normalization =
        CoefficientNormalization::Spherepack);

}  // namespace ylm::TensorYlm
