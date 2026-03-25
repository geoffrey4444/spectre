// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

namespace gr::Solutions {

/*!
 * \brief A perturbative Teukolsky wave on a flat background.
 *
 * \details This solution ports the core Cartesian metric perturbation formulas
 * from SpEC's linearized-gravity Teukolsky wave. It supports the
 * \f$l=2\f$ angular modes \f$m=-2,\dots,2\f$, even or odd parity, and ingoing
 * or outgoing propagation. The spatial metric is
 * \f$\gamma_{ij} = \delta_{ij} + h_{ij}\f$, with unit lapse and vanishing
 * shift. Since this is a perturbative wave solution, it should be used for
 * code-verification and extraction tests rather than as an exact solution of
 * the full nonlinear Einstein system.
 *
 * Spatial derivatives are supplied numerically from the analytic spatial
 * metric so the solution can be wrapped into generalized harmonic variables
 * for finite-radius extraction tests.
 */
class TeukolskyWave : public AnalyticSolution<3>,
                      public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = 3;

  struct Amplitude {
    using type = double;
    static constexpr Options::String help{"Amplitude of the perturbation"};
  };

  struct Mode {
    using type = int;
    static constexpr Options::String help{
        "Azimuthal mode m of the l=2 Teukolsky wave"};
    static type lower_bound() { return -2; }
    static type upper_bound() { return 2; }
  };

  struct Parity {
    using type = std::string;
    static constexpr Options::String help{
        "Parity of the perturbation: 'even' or 'odd'"};
  };

  struct Direction {
    using type = std::string;
    static constexpr Options::String help{
        "Propagation direction: 'outgoing' or 'ingoing'"};
  };

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help{
        "Center of the Teukolsky wave in inertial coordinates"};
  };

  struct Radius {
    using type = double;
    static constexpr Options::String help{
        "Radius of the center of the Gaussian pulse at t=0"};
  };

  struct Width {
    using type = double;
    static constexpr Options::String help{
        "Width of the Gaussian pulse profile"};
    static type lower_bound() { return 0.0; }
  };

  using options =
      tmpl::list<Amplitude, Mode, Parity, Direction, Center, Radius, Width>;
  static constexpr Options::String help{
      "A perturbative Teukolsky wave on a flat background"};

  TeukolskyWave(double amplitude, int mode, std::string parity,
                std::string direction, std::array<double, 3> center,
                double radius, double width,
                const Options::Context& context = {});

  TeukolskyWave() = default;
  TeukolskyWave(const TeukolskyWave& /*rhs*/) = default;
  TeukolskyWave& operator=(const TeukolskyWave& /*rhs*/) = default;
  TeukolskyWave(TeukolskyWave&& /*rhs*/) = default;
  TeukolskyWave& operator=(TeukolskyWave&& /*rhs*/) = default;
  ~TeukolskyWave() = default;

  explicit TeukolskyWave(CkMigrateMessage* msg);

  template <typename DataType>
  using tags = typename AnalyticSolution<3>::template tags<DataType>;

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, double t,
      tmpl::list<RequestedTags...> /*meta*/) const {
    const auto all_vars = all_variables(x, t);
    return {get<RequestedTags>(all_vars)...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  double amplitude() const { return amplitude_; }
  int mode() const { return mode_; }
  const std::string& parity() const { return parity_; }
  const std::string& direction() const { return direction_; }
  const std::array<double, 3>& center() const { return center_; }
  double radius() const { return radius_; }
  double width() const { return width_; }

 private:
  struct PointwiseData {
    std::array<std::array<double, 3>, 3> spatial_metric{};
    std::array<std::array<double, 3>, 3> dt_spatial_metric{};
  };

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<tags<DataType>> all_variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, double t) const;

  static PointwiseData pointwise_metric(double x, double y, double z, double t,
                                        double amplitude, int mode,
                                        bool even_parity, bool ingoing,
                                        const std::array<double, 3>& center,
                                        double radius, double width);

  static constexpr double finite_difference_step_factor_ = 1.0e-4;

  double amplitude_{1.0};
  int mode_{2};
  std::string parity_{"even"};
  std::string direction_{"outgoing"};
  std::array<double, 3> center_{{0.0, 0.0, 0.0}};
  double radius_{10.0};
  double width_{1.0};
};

bool operator==(const TeukolskyWave& lhs, const TeukolskyWave& rhs);
bool operator!=(const TeukolskyWave& lhs, const TeukolskyWave& rhs);

template <typename DataType>
size_t teukolsky_wave_number_of_points(
    const tnsr::I<DataType, 3, Frame::Inertial>& /*x*/) {
  return 1;
}

template <>
inline size_t teukolsky_wave_number_of_points(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  return get<0>(x).size();
}

template <typename DataType>
double teukolsky_wave_component(const DataType& x, const size_t /*s*/) {
  return x;
}

template <>
inline double teukolsky_wave_component(const DataVector& x, const size_t s) {
  return x[s];
}

template <typename DataType>
void teukolsky_wave_set_component(gsl::not_null<DataType*> result,
                                  const size_t /*s*/, const double value) {
  *result = value;
}

template <>
inline void teukolsky_wave_set_component(
    const gsl::not_null<DataVector*> result, const size_t s,
    const double value) {
  (*result)[s] = value;
}

template <typename DataType>
tuples::tagged_tuple_from_typelist<TeukolskyWave::tags<DataType>>
TeukolskyWave::all_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                             const double t) const {
  auto lapse = make_with_value<Scalar<DataType>>(get<0>(x), 1.0);
  auto dt_lapse = make_with_value<Scalar<DataType>>(get<0>(x), 0.0);
  auto deriv_lapse =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  auto shift =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  auto dt_shift =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  auto deriv_shift =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  auto dt_spatial_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  auto deriv_spatial_metric =
      make_with_value<tnsr::ijj<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);

  const bool even_parity = parity_ == "even";
  const bool ingoing = direction_ == "ingoing";
  const size_t number_of_points = teukolsky_wave_number_of_points(x);

  for (size_t s = 0; s < number_of_points; ++s) {
    const double x0 = teukolsky_wave_component(get<0>(x), s);
    const double x1 = teukolsky_wave_component(get<1>(x), s);
    const double x2 = teukolsky_wave_component(get<2>(x), s);
    const auto point_data =
        pointwise_metric(x0, x1, x2, t, amplitude_, mode_, even_parity, ingoing,
                         center_, radius_, width_);

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        teukolsky_wave_set_component(make_not_null(&spatial_metric.get(i, j)),
                                     s, point_data.spatial_metric[i][j]);
        teukolsky_wave_set_component(
            make_not_null(&dt_spatial_metric.get(i, j)), s,
            point_data.dt_spatial_metric[i][j]);
      }
    }

    const double centered_x = x0 - center_[0];
    const double centered_y = x1 - center_[1];
    const double centered_z = x2 - center_[2];
    const double local_radius =
        sqrt(centered_x * centered_x + centered_y * centered_y +
             centered_z * centered_z);
    const double step =
        finite_difference_step_factor_ * std::max(1.0, local_radius);
    for (size_t k = 0; k < 3; ++k) {
      const double dx0 = k == 0 ? step : 0.0;
      const double dx1 = k == 1 ? step : 0.0;
      const double dx2 = k == 2 ? step : 0.0;
      const auto plus_data =
          pointwise_metric(x0 + dx0, x1 + dx1, x2 + dx2, t, amplitude_, mode_,
                           even_parity, ingoing, center_, radius_, width_);
      const auto minus_data =
          pointwise_metric(x0 - dx0, x1 - dx1, x2 - dx2, t, amplitude_, mode_,
                           even_parity, ingoing, center_, radius_, width_);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          teukolsky_wave_set_component(
              make_not_null(&deriv_spatial_metric.get(k, i, j)), s,
              (plus_data.spatial_metric[i][j] -
               minus_data.spatial_metric[i][j]) /
                  (2.0 * step));
        }
      }
    }
  }

  const auto det_and_inverse = determinant_and_inverse(spatial_metric);
  auto sqrt_det_spatial_metric =
      make_with_value<Scalar<DataType>>(get<0>(x), 0.0);
  get(sqrt_det_spatial_metric) = sqrt(get(det_and_inverse.first));
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(get<0>(x), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      extrinsic_curvature.get(i, j) = -0.5 * dt_spatial_metric.get(i, j);
    }
  }

  return {std::move(lapse),
          std::move(dt_lapse),
          std::move(deriv_lapse),
          std::move(shift),
          std::move(dt_shift),
          std::move(deriv_shift),
          std::move(spatial_metric),
          std::move(dt_spatial_metric),
          std::move(deriv_spatial_metric),
          std::move(sqrt_det_spatial_metric),
          std::move(extrinsic_curvature),
          std::move(det_and_inverse.second)};
}

}  // namespace gr::Solutions
