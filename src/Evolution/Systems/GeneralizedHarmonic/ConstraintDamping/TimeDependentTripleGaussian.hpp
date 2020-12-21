// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace GeneralizedHarmonic::ConstraintDamping {
/*!
 * \brief A sum of three Gaussians plus a constant, where the Gaussian widths
 * are scaled by a domain::FunctionsOfTime::FunctionOfTime.
 *
 * \details The function \f$f\f$ is given by
 * \f{align}{
 * f = C + \sum_{\alpha=1}^3
 * A_\alpha \exp\left(-\frac{(x-(x_0)_\alpha)^2}{w_\alpha^2(t)}\right).
 * \f}
 * Input file options are: `Constant` \f$C\f$, `Amplitude[1-3]`
 * \f$A_\alpha\f$, `Width[1-3]` \f$w_\alpha\f$, `Center[1-3]
 * `\f$(x_0)_\alpha\f$, and `FunctionOfTimeForScaling`, a string naming a
 * domain::FunctionsOfTime::FunctionOfTime in the domain::Tags::FunctionsOfTime
 * that will be passed to the call operator. The function takes input
 * coordinates \f$x\f$ of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is e.g.
 * `double` or `DataVector`, `Fr` is a frame (e.g. `Frame::Inertial`), and
 * `VolumeDim` is the dimension of the spatial volume. The Gaussian widths
 * \f$w_\alpha\f$ are scaled by the inverse of the value of a scalar
 * domain::FunctionsOfTime::FunctionOfTime \f$f(t)\f$ named
 * `FunctionOfTimeForScaling`: \f$w_\alpha(t) = w_\alpha / f(t)\f$.
 */
template <size_t VolumeDim, typename Fr>
class TimeDependentTripleGaussian : public DampingFunction<VolumeDim, Fr> {
 public:
  struct Constant {
    using type = double;
    static constexpr Options::String help = {"The constant."};
  };

  struct Amplitude1 {
    using type = double;
    static constexpr Options::String help = {"The amplitudes of Gaussian 1."};
  };

  struct Width1 {
    using type = double;
    static constexpr Options::String help = {
        "The unscaled widths of Gaussian 1."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Center1 {
    using type = std::array<double, VolumeDim>;
    static constexpr Options::String help = {"The centers of Gaussian 1."};
  };

  struct Amplitude2 {
    using type = double;
    static constexpr Options::String help = {"The amplitudes of Gaussian 2."};
  };

  struct Width2 {
    using type = double;
    static constexpr Options::String help = {
        "The unscaled widths of Gaussian 2."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Center2 {
    using type = std::array<double, VolumeDim>;
    static constexpr Options::String help = {"The centers of Gaussian 2."};
  };

  struct Amplitude3 {
    using type = double;
    static constexpr Options::String help = {"The amplitudes of Gaussian 3."};
  };

  struct Width3 {
    using type = double;
    static constexpr Options::String help = {
        "The unscaled widths of Gaussian 3."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Center3 {
    using type = std::array<double, VolumeDim>;
    static constexpr Options::String help = {"The centers of Gaussian 3."};
  };

  struct FunctionOfTimeForScaling {
    using type = std::string;
    static constexpr Options::String help = {"The name of the FunctionOfTime."};
  };

  using options = tmpl::list<Constant, Amplitude1, Width1, Center1, Amplitude2,
                             Width2, Center2, Amplitude3, Width3, Center3,
                             FunctionOfTimeForScaling>;

  static constexpr Options::String help = {
      "Computes a sum of a constant and 3 Gaussians (each with its own "
      "amplitude, width, and coordinate center), with the Gaussian widths "
      "scaled by the inverse of a FunctionOfTime."};

  /// \cond
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(DampingFunction<VolumeDim, Fr>),
                                     TimeDependentTripleGaussian);  // NOLINT

  explicit TimeDependentTripleGaussian(CkMigrateMessage* /*unused*/) noexcept {}
  /// \endcond

  TimeDependentTripleGaussian(
      double constant, double amplitude_1, double width_1,
      const std::array<double, VolumeDim>& center_1, double amplitude_2,
      double width_2, const std::array<double, VolumeDim>& center_2,
      double amplitude_3, double width_3,
      const std::array<double, VolumeDim>& center_3,
      std::string function_of_time_for_scaling) noexcept;

  TimeDependentTripleGaussian() = default;
  ~TimeDependentTripleGaussian() override = default;
  TimeDependentTripleGaussian(const TimeDependentTripleGaussian& /*rhs*/) =
      default;
  TimeDependentTripleGaussian& operator=(
      const TimeDependentTripleGaussian& /*rhs*/) = default;
  TimeDependentTripleGaussian(TimeDependentTripleGaussian&& /*rhs*/) noexcept =
      default;
  TimeDependentTripleGaussian& operator=(
      TimeDependentTripleGaussian&& /*rhs*/) noexcept = default;

  void operator()(const gsl::not_null<Scalar<double>*> value_at_x,
                  const tnsr::I<double, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const noexcept override;
  void operator()(const gsl::not_null<Scalar<DataVector>*> value_at_x,
                  const tnsr::I<DataVector, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const noexcept override;

  auto get_clone() const noexcept
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  friend bool operator==(const TimeDependentTripleGaussian& lhs,
                         const TimeDependentTripleGaussian& rhs) noexcept {
    return lhs.constant_ == rhs.constant_ and
           lhs.amplitude_1_ == rhs.amplitude_1_ and
           lhs.inverse_width_1_ == rhs.inverse_width_1_ and
           lhs.center_1_ == rhs.center_1_ and
           lhs.amplitude_2_ == rhs.amplitude_2_ and
           lhs.inverse_width_2_ == rhs.inverse_width_2_ and
           lhs.center_2_ == rhs.center_2_ and
           lhs.amplitude_3_ == rhs.amplitude_3_ and
           lhs.inverse_width_3_ == rhs.inverse_width_3_ and
           lhs.center_3_ == rhs.center_3_ and
           lhs.function_of_time_for_scaling_ ==
               rhs.function_of_time_for_scaling_;
  }

  double constant_ = std::numeric_limits<double>::signaling_NaN();
  double amplitude_1_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_1_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, VolumeDim> center_1_{};
  double amplitude_2_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_2_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, VolumeDim> center_2_{};
  double amplitude_3_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_3_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, VolumeDim> center_3_{};
  std::string function_of_time_for_scaling_;

  template <typename T>
  void apply_call_operator(
      const gsl::not_null<Scalar<T>*> value_at_x,
      const tnsr::I<T, VolumeDim, Fr>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;
};

template <size_t VolumeDim, typename Fr>
bool operator!=(
    const TimeDependentTripleGaussian<VolumeDim, Fr>& lhs,
    const TimeDependentTripleGaussian<VolumeDim, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID GeneralizedHarmonic::ConstraintDamping::
    TimeDependentTripleGaussian<VolumeDim, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
