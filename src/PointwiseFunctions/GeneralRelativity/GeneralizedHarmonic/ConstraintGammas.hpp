// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
namespace Tags {
/*!
 * \brief Compute items to compute constraint-damping parameters for a
 * single-BH evolution.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::Tags::ConstraintGamma0`,
 * `GeneralizedHarmonic::Tags::ConstraintGamma1`, and
 * `GeneralizedHarmonic::Tags::ConstraintGamma2`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma0Compute : ConstraintGamma0, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    tnsr::I<DataVector, SpatialDim, Frame> centered_coords = coords;
    get<0>(centered_coords) -= 10.0;
    get(*gamma) = 3. * exp(-0.0078125 *
                           get(dot_product(centered_coords, centered_coords))) +
                  0.001;
  }

  using base = ConstraintGamma0;
};
/// \copydoc ConstraintGamma0Compute
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma1, get<0>(coords).size());
    get(*gamma1) = -1.;
  }

  using base = ConstraintGamma1;
};
/// \copydoc ConstraintGamma0Compute
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    tnsr::I<DataVector, SpatialDim, Frame> centered_coords = coords;
    get<0>(centered_coords) -= 10.0;
    get(*gamma) =
        exp(-0.0078125 * get(dot_product(centered_coords, centered_coords))) +
        0.001;
  }

  using base = ConstraintGamma2;
};

template <size_t SpatialDim, typename Frame>
struct ConstraintGamma0BBHCompute : ConstraintGamma0, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());

    constexpr double m_A = 0.5;
    constexpr double m_B = 0.5;
    constexpr double x_A = 10.0;
    constexpr double x_B = -10.0;

    constexpr double width_A = 7.0 / m_A;
    constexpr double width_B = 7.0 / m_B;
    constexpr double width_O = 2.5 * (x_A - x_B);
    constexpr double amp_A = 4.0 / m_A;
    constexpr double amp_B = 4.0 / m_B;
    constexpr double amp_O = 0.075 / (m_A + m_B);
    constexpr double asymptotic_damping = 0.001 / (m_A + m_B);

    auto distance_A_squared = make_with_value<Scalar<DataVector>>(coords, 0.0);
    get(distance_A_squared) = square(get<0>(coords) - x_A) +
                              square(get<1>(coords)) + square(get<2>(coords));
    auto distance_B_squared = make_with_value<Scalar<DataVector>>(coords, 0.0);
    get(distance_B_squared) = square(get<0>(coords) - x_B) +
                              square(get<1>(coords)) + square(get<2>(coords));
    auto distance_O_squared = make_with_value<Scalar<DataVector>>(coords, 0.0);
    get(distance_O_squared) = get(dot_product(coords, coords));

    get(*gamma) = amp_A * exp(-get(distance_A_squared) / square(width_A)) +
                  amp_B * exp(-get(distance_B_squared) / square(width_B)) +
                  amp_O * exp(-get(distance_O_squared) / square(width_O)) +
                  asymptotic_damping;
  }

  using base = ConstraintGamma0;
};

template <size_t SpatialDim, typename Frame>
struct ConstraintGamma1BBHCompute : ConstraintGamma1, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma1, get<0>(coords).size());
    constexpr double amp = 0.999;
    constexpr double x_A = 10.0;
    constexpr double x_B = -10.0;

    constexpr double width = 10.0 * (x_A - x_B);
    get(*gamma1) =
        amp * (-1.0 + exp(-get(dot_product(coords, coords)) / width));
  }

  using base = ConstraintGamma1;
};

template <size_t SpatialDim, typename Frame>
struct ConstraintGamma2BBHCompute : ConstraintGamma2, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    constexpr double m_A = 0.5;
    constexpr double m_B = 0.5;
    constexpr double x_A = 10.0;
    constexpr double x_B = -10.0;

    constexpr double width_A = 7.0 / m_A;
    constexpr double width_B = 7.0 / m_B;
    constexpr double width_O = 2.5 * (x_A - x_B);
    constexpr double amp_A = 4.0 / m_A;
    constexpr double amp_B = 4.0 / m_B;
    constexpr double amp_O = 0.075 / (m_A + m_B);
    constexpr double asymptotic_damping = 0.001 / (m_A + m_B);

    auto distance_A_squared = make_with_value<Scalar<DataVector>>(coords, 0.0);
    get(distance_A_squared) = square(get<0>(coords) - x_A) +
                              square(get<1>(coords)) + square(get<2>(coords));
    auto distance_B_squared = make_with_value<Scalar<DataVector>>(coords, 0.0);
    get(distance_B_squared) = square(get<0>(coords) - x_B) +
                              square(get<1>(coords)) + square(get<2>(coords));
    auto distance_O_squared = make_with_value<Scalar<DataVector>>(coords, 0.0);
    get(distance_O_squared) = get(dot_product(coords, coords));

    get(*gamma) = amp_A * exp(-get(distance_A_squared) / square(width_A)) +
                  amp_B * exp(-get(distance_B_squared) / square(width_B)) +
                  amp_O * exp(-get(distance_O_squared) / square(width_O)) +
                  asymptotic_damping;
  }

  using base = ConstraintGamma2;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
