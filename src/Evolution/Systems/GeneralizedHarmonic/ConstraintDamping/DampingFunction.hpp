// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"

/// \cond
class DataVector;
/// \endcond

/// Holds classes implementing DampingFunction (functions \f$R^n \to R\f$).
namespace GeneralizedHarmonic::ConstraintDamping {
/// \cond
template <size_t VolumeDim, typename Fr>
class GaussianPlusConstant;
template <size_t VolumeDim, typename Fr>
class TripleGaussianPlusConstant;
/// \endcond

/*!
 * \brief Base class defining interface for constraint damping functions.
 *
 * Encodes a function \f$R^n \to R\f$ where n is `VolumeDim` that represents
 * a generalized-harmonic constraint-damping parameter (i.e., Gamma0,
 * Gamma1, or Gamma2).
 *
 * By default, a `DampingFunction` is time independent. A `DampingFunction` can
 * have a simple form of time dependence, in which a single, time-dependent
 * member variable (a double) is updated (e.g., to track the current value of a
 * scalar FunctionOfTime, such as the expansion factor in a compact binary
 * inspiral). To implement a time-dependent `DampingFunction`, set its
 * is_time_dependent == true, set its function_of_time_for_scaling_name to the
 * name of a scalar FunctionOfTime. When using a time-dependent
 * `DampingFunction`, set member variable time_dependent_scale to this
 * FunctionOfTime's current value before calling the call operator for the first
 * time and before calling the call operator after time has changed.
 */
template <size_t VolumeDim, typename Fr>
class DampingFunction : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<GeneralizedHarmonic::ConstraintDamping::GaussianPlusConstant<
                     VolumeDim, Fr>,
                 GeneralizedHarmonic::ConstraintDamping::
                     TripleGaussianPlusConstant<VolumeDim, Fr>>;
  constexpr static size_t volume_dim = VolumeDim;
  using frame = Fr;

  WRAPPED_PUPable_abstract(DampingFunction);  // NOLINT

  DampingFunction() = default;
  DampingFunction(const DampingFunction& /*rhs*/) = default;
  DampingFunction& operator=(const DampingFunction& /*rhs*/) = default;
  DampingFunction(DampingFunction&& /*rhs*/) noexcept = default;
  DampingFunction& operator=(DampingFunction&& /*rhs*/) noexcept = default;
  ~DampingFunction() override = default;

  //@{
  /// Returns the value of the function at the coordinate 'x'.
  virtual Scalar<double> operator()(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept = 0;
  virtual Scalar<DataVector> operator()(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept = 0;
  //@}

  virtual auto get_clone() const noexcept
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> = 0;

  // By default, a DampingFunction is not time dependent. To implement
  // a time-dependent DampingFunction, set is_time_dependent to true, and
  // set function_of_time_for_scaling_name to the name of a scalar
  // FunctionOfTime. Before calling the call operator for the
  // first time and whenever the time has changed, set member variable
  // `time_dependent_scale` to the current value of the `FunctionOfTime` whose
  // name is `function_of_time_for_scaling_name` before calling the call
  // operator.
  const static bool is_time_dependent = false;
  std::string function_of_time_for_scaling_name = ""s;
  double time_dependent_scale = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace GeneralizedHarmonic::ConstraintDamping

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/GaussianPlusConstant.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TripleGaussianPlusConstant.hpp"
