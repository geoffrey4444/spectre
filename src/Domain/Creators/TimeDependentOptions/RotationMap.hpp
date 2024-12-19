// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Class to be used as an option for initializing rotation map
 * coefficients.
 */
template <bool AllowSettleFoTs>
struct RotationMapOptions {
  static constexpr Options::String help = {
      "Options for a time-dependent rotation of the coordinates."};

  struct InitialQuaternions {
    using type = std::vector<std::array<double, 4>>;
    static constexpr Options::String help = {
        "Initial values for the quaternion of the rotation map. You can "
        "optionally specify its first two time derivatives. If time "
        "derivatives aren't specified, zero will be used."};
  };

  struct InitialAngles {
    using type = std::vector<std::array<double, 3>>;
    static constexpr Options::String help = {
        "Initial values for the angle of the rotation map. You can optionally "
        "specify its first two time derivatives (angular velocity and "
        "acceleration). If time derivatives aren't specified, zero will be "
        "used."};
  };

  struct DecayTimescale {
    using type = double;
    static constexpr Options::String help = {
        "The timescale for how fast the rotation approaches its asymptotic "
        "value. If this is specified, a SettleToConstant function of time will "
        "be used. If 'Auto' is specified, a PiecewisePolynomial function of "
        "time will be used."};
  };

  using options = tmpl::conditional_t<
      AllowSettleFoTs,
      tmpl::list<InitialQuaternions,
                 Options::Alternatives<tmpl::list<InitialAngles>,
                                       tmpl::list<DecayTimescale>>>,
      tmpl::list<InitialQuaternions, InitialAngles>>;

  RotationMapOptions() = default;
  // Constructor tha can handle everything
  RotationMapOptions(
      const std::vector<std::array<double, 4>>& initial_quaternions,
      const std::vector<std::array<double, 3>>& initial_angles,
      std::optional<double> decay_timescale_in,
      const Options::Context& context = {});
  // Constructor for non SettleToConstant functions of time
  RotationMapOptions(
      const std::vector<std::array<double, 4>>& initial_quaternions,
      const std::vector<std::array<double, 3>>& initial_angles,
      const Options::Context& context = {})
      : RotationMapOptions(initial_quaternions, initial_angles, std::nullopt,
                           context) {}
  // Constructor for SettleToConst functions of time
  RotationMapOptions(
      const std::vector<std::array<double, 4>>& initial_quaternions,
      double decay_timescale_in, const Options::Context& context = {})
      : RotationMapOptions(initial_quaternions,
                           std::vector{std::array{0.0, 0.0, 0.0}},
                           {decay_timescale_in}, context) {}

  std::array<DataVector, 3> quaternions{};
  std::array<DataVector, 4> angles{};
  std::optional<double> decay_timescale;
};

template <bool AllowSettleFoTs>
struct RotationMap {
  using type = Options::Auto<
      std::variant<RotationMapOptions<AllowSettleFoTs>, FromVolumeFile>,
      Options::AutoLabel::None>;
  static constexpr Options::String help = {
      "Options for a time-dependent rotation of the coordinates. Specify "
      "'None' to not use this map."};
};

template <bool AllowSettleFoTs>
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> set_rotation(
    const std::variant<RotationMapOptions<AllowSettleFoTs>, FromVolumeFile>&
        rotation_map_options,
    double initial_time, double expiration_time);
}  // namespace domain::creators::time_dependent_options
