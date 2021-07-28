// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <size_t Dim>
class Wedge;
template <size_t VolumeDim>
class Identity;
namespace TimeDependent {
template <size_t VolumeDim>
class Rotation;
}  // namespace TimeDependent
}  // namespace CoordinateMaps
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Disk_detail {
// If `Metavariables` has a `domain_parameters` member struct and
// `domain_parameters::enable_time_dependent_maps` is `true`, then
// inherit from `std::true_type`; otherwise, inherit from `std::false_type`.
template <typename Metavariables, typename = std::void_t<>>
struct enable_time_dependent_maps : std::false_type {};

template <typename Metavariables>
struct enable_time_dependent_maps<Metavariables,
                                  std::void_t<typename Metavariables::domain>>
    : std::bool_constant<Metavariables::domain::enable_time_dependent_maps> {};

template <typename Metavariables>
constexpr bool enable_time_dependent_maps_v =
    enable_time_dependent_maps<Metavariables>::value;
}  // namespace Disk_detail
/// \endcond

namespace domain {
namespace creators {
/*!
 * Create a 2D Domain in the shape of a disk from a square surrounded by four
 * wedges.
 *
 * \note When using this domain, the
 * metavariables struct can contain a struct named `domain`
 * that conforms to domain::protocols::Metavariables. If
 * domain::enable_time_dependent_maps is either set to `false`
 * or not specified in the metavariables, then this domain will be
 * time-independent. If domain::enable_time_dependent_maps is set
 * to `true`, then this domain also includes a time-dependent map, along with
 * additional options (and a corresponding constructor) for initializing the
 * time-dependent map. These options include `InitialTime` and
 * `InitialExpirationDeltaT`, which specify the initial time and the
 * initial updating time interval, respectively, for the FunctionsOfTime
 * controlling the map. The time-dependent map itself consists of a Rotation map
 * about the z axis.
 */
class Disk : public DomainCreator<2> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<
                     Frame::Logical, Frame::Inertial,
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                    CoordinateMaps::Affine>>,
                 domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       CoordinateMaps::ProductOf2Maps<
                                           CoordinateMaps::Equiangular,
                                           CoordinateMaps::Equiangular>>,
                 domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       CoordinateMaps::Wedge<2>>,
                 domain::CoordinateMap<
                     Frame::Grid, Frame::Inertial,
                     domain::CoordinateMaps::TimeDependent::Rotation<2>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the circle circumscribing the inner square."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the Disk."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,theta]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() noexcept { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "The boundary condition to impose on all sides.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  // The following options are for optional time dependent maps
  struct TimeDependentMaps {
    static constexpr Options::String help = {"Options for time-dependent maps"};
  };

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
    using group = TimeDependentMaps;
  };
  /// \brief The initial time interval for updates of the functions of time.
  struct InitialExpirationDeltaT {
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "The initial time interval for updates of the functions of time. If "
        "Auto, then the functions of time do not expire, nor can they be "
        "updated."};
    using group = TimeDependentMaps;
  };

  struct RotationAboutZAxisMap {
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map about the z axis"};
    using group = TimeDependentMaps;
  };
  /// \brief The initial value of the rotation angle.
  struct InitialRotationAngle {
    using type = double;
    static constexpr Options::String help = {"Rotation angle at initial time."};
    using group = RotationAboutZAxisMap;
  };
  /// \brief The angular velocity of the rotation.
  struct InitialAngularVelocity {
    using type = double;
    static constexpr Options::String help = {"The angular velocity."};
    using group = RotationAboutZAxisMap;
  };
  /// \brief The name of the function of time to be added to the added to the
  /// DataBox for the rotation-about-the-z-axis map.
  struct RotationAboutZAxisFunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {"Name of the function of time."};
    using group = RotationAboutZAxisMap;
    static std::string name() noexcept { return "FunctionOfTimeName"; }
  };

  using basic_options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                                   InitialGridPoints, UseEquiangularMap>;

  template <typename Metavariables>
  using time_independent_options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          BoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  using time_dependent_options =
      tmpl::list<InitialTime, InitialExpirationDeltaT, InitialRotationAngle,
                 InitialAngularVelocity, RotationAboutZAxisFunctionOfTimeName>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      Disk_detail::enable_time_dependent_maps_v<Metavariables>,
      tmpl::append<time_dependent_options,
                   time_independent_options<Metavariables>>,
      time_independent_options<Metavariables>>;

  static constexpr Options::String help{
      "Creates a 2D Disk with five Blocks.\n"
      "Only one refinement level for both dimensions is currently supported.\n"
      "The number of gridpoints in each dimension can be set independently.\n"
      "The number of gridpoints along the dimensions of the square is equal\n"
      "to the number of gridpoints along the angular dimension of the wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block. Note that the domain optionally includes "
      "a time-dependent rotation map; enabling "
      "the time-dependent map requires adding a "
      "struct named domain to the Metavariables, with this "
      "struct conforming to domain::protocols::Metavariables. To enable the "
      "time-dependent map, set "
      "Metavariables::domain::enable_time_dependent_maps to "
      "true."};

  // Constructor for time-independent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == false or
  // when the metavariables do not define
  // Metavariables::domain::enable_time_dependent_maps)
  Disk(typename InnerRadius::type inner_radius,
       typename OuterRadius::type outer_radius,
       typename InitialRefinement::type initial_refinement,
       typename InitialGridPoints::type initial_number_of_grid_points,
       typename UseEquiangularMap::type use_equiangular_map,
       std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
           boundary_condition = nullptr,
       const Options::Context& context = {});

  // Constructor for time-dependent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == true),
  // with parameters corresponding to the additional options
  Disk(double initial_time, std::optional<double> initial_expiration_delta_t,
       double initial_rotation_angle, double initial_angular_velocity,
       std::string rotation_about_z_axis_function_of_time_name,
       typename InnerRadius::type inner_radius,
       typename OuterRadius::type outer_radius,
       typename InitialRefinement::type initial_refinement,
       typename InitialGridPoints::type initial_number_of_grid_points,
       typename UseEquiangularMap::type use_equiangular_map,
       std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
           boundary_condition = nullptr,
       const Options::Context& context = {});

  Disk() = default;
  Disk(const Disk&) = delete;
  Disk(Disk&&) noexcept = default;
  Disk& operator=(const Disk&) = delete;
  Disk& operator=(Disk&&) noexcept = default;
  ~Disk() noexcept override = default;

  Domain<2> create_domain() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const
      noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  void check_for_parse_errors(const Options::Context& context) const;

  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_{false};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;

  // Variables for FunctionsOfTime options
  bool enable_time_dependence_;
  double initial_time_;
  std::optional<double> initial_expiration_delta_t_;
  double initial_rotation_angle_;
  double initial_angular_velocity_;
  std::string rotation_about_z_axis_function_of_time_name_;
};
}  // namespace creators
}  // namespace domain
