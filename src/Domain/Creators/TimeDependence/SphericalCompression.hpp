// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
namespace CoordinateMaps::TimeDependent {
template <bool InertiorMap>
class SphericalCompression;
}  // namespace CoordinateMaps::TimeDependent
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::time_dependence {
/*!
 * \brief A spherical compression about some center, as given by
 * domain::CoordinateMaps::TimeDependent::SphericalCompression<false>.
 *
 * \details This TimeDependence is suitable for use on a spherical shell,
 * where MinRadius and MaxRadius are the inner and outer radii of the shell,
 * respectively.
 */
template <size_t MeshDim>
class SphericalCompression final : public TimeDependence<MeshDim> {
  static_assert(MeshDim == 3,
                "SphericalCompression time dependence only defined for 3 "
                "spatial dimensions");

 private:
  using SphericalCompressionMap =
      domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;

 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       SphericalCompressionMap>>;

  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the function of time"};
  };
  /// \brief The time interval for updates of the functions of time.
  struct InitialExpirationDeltaT {
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "The initial time interval for updates of the functions of time. If "
        "Auto, then the functions of time do not expire, nor can they be "
        "updated."};
  };
  /// \brief Minimum radius for the SphericalCompression map
  struct MinRadius {
    using type = double;
    static constexpr Options::String help = {
        "Min radius for spherical compression map."};
  };
  /// \brief Maximum radius for the SphericalCompression map
  struct MaxRadius {
    using type = double;
    static constexpr Options::String help = {
        "Max radius for spherical compression map."};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center for the spherical compression map."};
  };
  /// \brief Initial value for function of time for the size map
  struct InitialValue {
    using type = double;
    static constexpr Options::String help = {"SizeMap value at initial time."};
  };
  /// \brief Initial velocity for the function of time for the size map
  struct InitialVelocity {
    using type = double;
    static constexpr Options::String help = {"SizeMap initial velocity."};
  };
  /// \brief Initial acceleration for the function of time for the size map
  struct InitialAcceleration {
    using type = double;
    static constexpr Options::String help = {"SizeMap initial acceleration."};
  };
  /// \brief The name of the functions of time to be added to the added to the
  /// DataBox for the size map.
  struct FunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {
        "Names of SizeMap function of time."};
  };

  using MapForComposition =
      detail::generate_coordinate_map_t<tmpl::list<SphericalCompressionMap>>;

  using options = tmpl::list<InitialTime, InitialExpirationDeltaT, MinRadius,
                             MaxRadius, Center, InitialValue, InitialVelocity,
                             InitialAcceleration, FunctionOfTimeName>;

  static constexpr Options::String help = {"A spherical compression."};

  SphericalCompression() = default;
  ~SphericalCompression() override = default;
  SphericalCompression(const SphericalCompression&) = delete;
  SphericalCompression(SphericalCompression&&) noexcept = default;
  SphericalCompression& operator=(const SphericalCompression&) = delete;
  SphericalCompression& operator=(SphericalCompression&&) noexcept = default;

  SphericalCompression(
      double initial_time, std::optional<double> initial_expiration_delta_t,
      double min_radius, double max_radius, std::array<double, 3> center,
      double initial_value, double initial_velocity,
      double initial_acceleration,
      std::string function_of_time_name = "LambdaFactorA0") noexcept;

  auto get_clone() const noexcept
      -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps(size_t number_of_blocks) const noexcept
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  /// Returns the map for each block to be used in a composition of
  /// `TimeDependence`s.
  MapForComposition map_for_composition() const noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const SphericalCompression<LocalDim>& lhs,
                         const SphericalCompression<LocalDim>& rhs) noexcept;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  std::optional<double> initial_expiration_delta_t_{};
  double min_radius_{std::numeric_limits<double>::signaling_NaN()};
  double max_radius_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center_{};
  double initial_value_{std::numeric_limits<double>::signaling_NaN()};
  double initial_velocity_{std::numeric_limits<double>::signaling_NaN()};
  double initial_acceleration_{std::numeric_limits<double>::signaling_NaN()};
  std::string function_of_time_name_{};
};

template <size_t Dim>
bool operator==(const SphericalCompression<Dim>& lhs,
                const SphericalCompression<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const SphericalCompression<Dim>& lhs,
                const SphericalCompression<Dim>& rhs) noexcept;
}  // namespace domain::creators::time_dependence
