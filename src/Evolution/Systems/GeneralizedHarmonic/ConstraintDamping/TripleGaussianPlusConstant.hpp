// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>

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
/// \endcond

namespace GeneralizedHarmonic::ConstraintDamping {
/*!
 * \brief A sum of three Gaussians and a constant: \f$f = C + \sum_{n=0}^2 A_n
 * \exp\left(-\frac{(x-x_n)^2}{w_n^2}\right)\f$. The Gaussian widths are
 * scaled time dependently (e.g., by the expansion factor in a compact
 * binary evolution).
 *
 * \details Input file options are as follows: `Constant` \f$C\f$; Gaussian
 * amplitudes `Amplitudes` \f[$A_0, A_1, A_2]\f$, Gaussian widths (before a
 * time-dependent scaling is applied) `Widths` \f$[w_10 w_1, $w_2\f$, centers
 * for each Gaussian `Centers` \f$[x_0, x_1, x_2]\f$, and `FunctionOfTimeName`,
 * which is the name of a scalar FunctionOfTime (e.g., `"ExpansionFactor"` that
 * should be used when rescaling the Gaussian widths). The function takes input
 * coordinates of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is e.g. `double`
 * or `DataVector`, `Fr` is a frame (e.g. `Frame::Inertial`), and `VolumeDim` is
 * the dimension of the spatial volume. Before calling the call operator for the
 * first time and whenever the time has changed, set member variable
 * `time_dependent_scale` to the current value of the `FunctionOfTime`
 * `FunctionofTimeName` before calling the call operator.
 */
template <size_t VolumeDim, typename Fr>
class TripleGaussianPlusConstant : public DampingFunction<VolumeDim, Fr> {
 public:
  struct Constant {
    using type = double;
    static constexpr Options::String help = {"The constant."};
  };

  struct Amplitudes {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Gaussian amplitudes [A0, A1, A2]."};
  };

  struct Widths {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"Gaussian widths [w0, w1, w2]."};
    static type lower_bound() noexcept { return {{0., 0., 0.}}; }
  };

  struct Centers {
    using type = std::array<std::array<double, VolumeDim>, 3>;
    static constexpr Options::String help = {
        "Gaussian centers [[x_i, y_i, z_i]] (i from 0 to 2)."};
  };

  struct FunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {
        "Name of a scalar FunctionOfTime for rescaling Gaussian widths"};
  };
  using options =
      tmpl::list<Constant, Amplitudes, Widths, Centers, FunctionOfTimeName>;

  static constexpr Options::String help = {
      "Computes a sum of three Gaussians and a constant. Each Gaussian is "
      "specified by an amplitude, width, and center. The Gaussian widths have "
      "a time-dependent rescaling applied based on a scalar FunctionOfTime."};

  /// \cond
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(DampingFunction<VolumeDim, Fr>),
                                     TripleGaussianPlusConstant);  // NOLINT

  explicit TripleGaussianPlusConstant(CkMigrateMessage* /*unused*/) noexcept {}
  /// \endcond

  TripleGaussianPlusConstant(
      double constant, const std::array<double, 3>& amplitudes,
      const std::array<double, 3>& widths,
      const std::array<std::array<double, VolumeDim>, 3>& centers,
      const std::string& function_of_time_name) noexcept;

  TripleGaussianPlusConstant() = default;
  ~TripleGaussianPlusConstant() override = default;
  TripleGaussianPlusConstant(const TripleGaussianPlusConstant& /*rhs*/) =
      default;
  TripleGaussianPlusConstant& operator=(
      const TripleGaussianPlusConstant& /*rhs*/) = default;
  TripleGaussianPlusConstant(TripleGaussianPlusConstant&& /*rhs*/) noexcept =
      default;
  TripleGaussianPlusConstant& operator=(
      TripleGaussianPlusConstant&& /*rhs*/) noexcept = default;

  Scalar<double> operator()(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  Scalar<DataVector> operator()(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  auto get_clone() const noexcept
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

  // time dependence
  const static bool is_time_dependent = true;

 private:
  friend bool operator==(const TripleGaussianPlusConstant& lhs,
                         const TripleGaussianPlusConstant& rhs) noexcept {
    return lhs.constant_ == rhs.constant_ and
           lhs.amplitudes_ == rhs.amplitudes_ and
           lhs.inverse_widths_ == rhs.inverse_widths_ and
           lhs.centers_ == rhs.centers_ and
           lhs.function_of_time_for_scaling_name ==
               rhs.function_of_time_for_scaling_name;
  }

  double constant_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> amplitudes_{};
  std::array<double, 3> inverse_widths_{};
  std::array<std::array<double, VolumeDim>, 3> centers_{};

  template <typename T>
  tnsr::I<T, VolumeDim, Fr> centered_coordinates(
      const tnsr::I<T, VolumeDim, Fr>& x, size_t which_gaussian) const noexcept;

  template <typename T>
  Scalar<T> apply_call_operator(
      const tnsr::I<T, VolumeDim, Fr>& centered_coords_0,
      const tnsr::I<T, VolumeDim, Fr>& centered_coords_1,
      const tnsr::I<T, VolumeDim, Fr>& centered_coords_2) const noexcept;
};

template <size_t VolumeDim, typename Fr>
bool operator!=(const TripleGaussianPlusConstant<VolumeDim, Fr>& lhs,
                const TripleGaussianPlusConstant<VolumeDim, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID GeneralizedHarmonic::ConstraintDamping::
    TripleGaussianPlusConstant<VolumeDim, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
