// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace GeneralizedHarmonic::ConstraintDamping {
/// \cond
template <size_t VolumeDim, typename Fr>
class DampingFunction;
/// \endcond

/*!
 * \brief A struct that stores the generalized harmonic constraint damping
 * parameters
 *
 */
template <size_t VolumeDim, typename Fr>
struct DampingFunctions {
  std::unique_ptr<DampingFunction<VolumeDim, Fr>*> gamma0;
  std::unique_ptr<DampingFunction<VolumeDim, Fr>*> gamma1;
  std::unique_ptr<DampingFunction<VolumeDim, Fr>*> gamma2;

  static constexpr Options::String help{
      "A struct holding the DampingFunctions for the generalized harmonic "
      "cosntraint damping parameters."};

  struct DampingFunctionGamma0 {
    using type = std::unique_ptr<DampingFunction<VolumeDim, Fr>*>;
    static constexpr Options::String help{"DampingFunction for Gamma0"};
  };

  struct DampingFunctionGamma1 {
    using type = std::unique_ptr<DampingFunction<VolumeDim, Fr>*>;
    static constexpr Options::String help{"DampingFunction for Gamma1"};
  };

  struct DampingFunctionGamma2 {
    using type = std::unique_ptr<DampingFunction<VolumeDim, Fr>*>;
    static constexpr Options::String help{"DampingFunction for Gamma2"};
  };

  using options = tmpl::list<DampingFunctionGamma0, DampingFunctionGamma1,
                             DampingFunctionGamma2>;

  DampingFunctions(
      std::unique_ptr<DampingFunction<VolumeDim, Fr>*> damping_function_gamma0,
      std::unique_ptr<DampingFunction<VolumeDim, Fr>*> damping_function_gamma1,
      std::unique_ptr<DampingFunction<VolumeDim, Fr>*>
          damping_function_gamma2) noexcept;

  DampingFunctions() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT
};
}  // namespace GeneralizedHarmonic::ConstraintDamping
