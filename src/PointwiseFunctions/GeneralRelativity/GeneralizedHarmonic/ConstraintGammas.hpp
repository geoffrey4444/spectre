// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
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

namespace GeneralizedHarmonic::ConstraintDamping::Tags {
/*!
 * \brief Computes the constraint damping parameter \f$\gamma_0\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma0`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma0Compute : ConstraintGamma0, db::ComputeTag {
  using argument_tags =
      tmpl::list<DampingFunctionGamma0<SpatialDim, Frame>, ::Tags::Time,
                 ::domain::Tags::FunctionsOfTime,
                 domain::Tags::Coordinates<SpatialDim, Frame>>;
  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
          SpatialDim, Frame>& damping_function,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    if (damping_function.is_time_dependent) {
      damping_function.time_dependent_scale =
          gsl::at(functions_of_time,
                  damping_function.function_of_time_scaling_name)
              ->func(time)[0][0];
    }
    get(*gamma) = get(damping_function(coords));
  }

  using base = ConstraintGamma0;
};

/*!
 * \brief Computes the constraint damping parameter \f$\gamma_1\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags =
      tmpl::list<DampingFunctionGamma1<SpatialDim, Frame>, ::Tags::Time,
                 ::domain::Tags::FunctionsOfTime,
                 domain::Tags::Coordinates<SpatialDim, Frame>>;
  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
          SpatialDim, Frame>& damping_function,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma1, get<0>(coords).size());
    if (damping_function.is_time_dependent) {
      damping_function.time_dependent_scale =
          gsl::at(functions_of_time,
                  damping_function.function_of_time_scaling_name)
              ->func(time)[0][0];
    }
    get(*gamma1) = get(damping_function(coords));
  }

  using base = ConstraintGamma1;
};

/*!
 * \brief Computes the constraint damping parameter \f$\gamma_2\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags =
      tmpl::list<DampingFunctionGamma2<SpatialDim, Frame>, ::Tags::Time,
                 ::domain::Tags::FunctionsOfTime,
                 domain::Tags::Coordinates<SpatialDim, Frame>>;
  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
          SpatialDim, Frame>& damping_function,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    if (damping_function.is_time_dependent) {
      damping_function.time_dependent_scale =
          gsl::at(functions_of_time,
                  damping_function.function_of_time_scaling_name)
              ->func(time)[0][0];
    }
    get(*gamma) = get(damping_function(coords));
  }

  using base = ConstraintGamma2;
};
}  // namespace GeneralizedHarmonic::ConstraintDamping::Tags
