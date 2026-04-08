// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmTransforms.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <typename TensorType>
void fill_random_coefficients(const gsl::not_null<TensorType*> coefficients,
                              const size_t ell_max,
                              const size_t number_of_offsets) {
  const ylm::SpherepackIterator it(ell_max, ell_max, 1, false);
  const size_t spectral_size = it.spherepack_array_size();
  *coefficients = make_with_value<TensorType>(*coefficients, 0.0);

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  for (auto& component : *coefficients) {
    component.destructive_resize(spectral_size * number_of_offsets);
    for (ylm::SpherepackIterator coeff_it(ell_max, ell_max, 1, false); coeff_it;
         ++coeff_it) {
      const bool keep_mode =
          coeff_it.l() < ell_max + 1 - TensorType::rank() and
          (coeff_it.coefficient_array() !=
               ylm::SpherepackIterator::CoefficientArray::b or
           coeff_it.m() != 0);
      for (size_t offset = 0; offset < number_of_offsets; ++offset) {
        component[coeff_it() * number_of_offsets + offset] =
            keep_mode ? dist(generator) : 0.0;
      }
    }
  }
}

template <typename TensorType>
void check_transform_roundtrip(
    const size_t ell_max, const size_t number_of_offsets,
    const ylm::TensorYlm::CoefficientNormalization normalization) {
  const ylm::SpherepackIterator it(ell_max, ell_max, 1, false);
  const size_t total_size = it.spherepack_array_size() * number_of_offsets;
  auto coefficients = TensorType{};
  if constexpr (TensorType::rank() == 0) {
    coefficients.destructive_resize(total_size);
  } else {
    coefficients = TensorType(total_size);
  }
  fill_random_coefficients(make_not_null(&coefficients), ell_max,
                           number_of_offsets);
  const auto transformed = ylm::TensorYlm::scalar_to_tensor_ylm_coefficients(
      coefficients, ell_max, number_of_offsets, normalization);
  const auto recovered = ylm::TensorYlm::tensor_to_scalar_ylm_coefficients(
      transformed, ell_max, number_of_offsets, normalization);

  if constexpr (TensorType::rank() == 0) {
    CHECK_ITERABLE_CUSTOM_APPROX(recovered, coefficients, approx);
  } else {
    for (size_t component = 0; component < coefficients.size(); ++component) {
      CHECK_ITERABLE_CUSTOM_APPROX(recovered[component],
                                   coefficients[component], approx);
    }
  }
}

template <typename TensorType>
void check_multi_offset_matches_single_offset(
    const size_t ell_max,
    const ylm::TensorYlm::CoefficientNormalization normalization) {
  const size_t number_of_offsets = 3;
  const ylm::SpherepackIterator it(ell_max, ell_max, 1, false);
  const size_t total_size = it.spherepack_array_size() * number_of_offsets;
  auto coefficients = TensorType{};
  if constexpr (TensorType::rank() == 0) {
    coefficients.destructive_resize(total_size);
  } else {
    coefficients = TensorType(total_size);
  }
  fill_random_coefficients(make_not_null(&coefficients), ell_max,
                           number_of_offsets);

  const auto multi = ylm::TensorYlm::scalar_to_tensor_ylm_coefficients(
      coefficients, ell_max, number_of_offsets, normalization);
  for (size_t offset = 0; offset < number_of_offsets; ++offset) {
    auto single_input = TensorType{};
    if constexpr (TensorType::rank() == 0) {
      single_input.destructive_resize(it.spherepack_array_size());
      for (size_t i = 0; i < it.spherepack_array_size(); ++i) {
        single_input[i] = coefficients[i * number_of_offsets + offset];
      }
    } else {
      single_input = TensorType(it.spherepack_array_size());
      for (size_t component = 0; component < coefficients.size(); ++component) {
        for (size_t i = 0; i < it.spherepack_array_size(); ++i) {
          single_input[component][i] =
              coefficients[component][i * number_of_offsets + offset];
        }
      }
    }

    const auto single = ylm::TensorYlm::scalar_to_tensor_ylm_coefficients(
        single_input, ell_max, 1, normalization);
    if constexpr (TensorType::rank() == 0) {
      for (size_t i = 0; i < it.spherepack_array_size(); ++i) {
        CHECK(multi[i * number_of_offsets + offset] == approx(single[i]));
      }
    } else {
      for (size_t component = 0; component < coefficients.size(); ++component) {
        for (size_t i = 0; i < it.spherepack_array_size(); ++i) {
          CHECK(multi[component][i * number_of_offsets + offset] ==
                approx(single[component][i]));
        }
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.TensorYlmTransforms",
                  "[Unit][NumericalAlgorithms]") {
  check_transform_roundtrip<tnsr::i<DataVector, 3>>(
      5, 2, ylm::TensorYlm::CoefficientNormalization::Spherepack);
  check_transform_roundtrip<tnsr::ii<DataVector, 3>>(
      5, 2, ylm::TensorYlm::CoefficientNormalization::Spherepack);
  check_transform_roundtrip<tnsr::ijj<DataVector, 3>>(
      5, 2, ylm::TensorYlm::CoefficientNormalization::Standard);

  check_multi_offset_matches_single_offset<tnsr::i<DataVector, 3>>(
      4, ylm::TensorYlm::CoefficientNormalization::Spherepack);
  check_multi_offset_matches_single_offset<tnsr::ii<DataVector, 3>>(
      4, ylm::TensorYlm::CoefficientNormalization::Standard);
}
