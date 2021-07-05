// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

// IWYU wants to include things we definitely don't need...
// IWYU pragma: no_include <memory> // Needed in cpp file
// IWYU pragma: no_include <pup.h>  // Not needed

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp" // Not needed

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class EquatorialCompression;
class Equiangular;
template <size_t VolumeDim>
class Identity;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
namespace TimeDependent {
template <size_t VolumeDim>
class CubicScale;
template <size_t VolumeDim>
class Rotation;
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace TimeDependent
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
namespace Sphere_detail {
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
}  // namespace Sphere_detail
/// \endcond

namespace domain {
namespace creators {
/// Create a 3D Domain in the shape of a sphere consisting of six wedges
/// and a central cube. For an image showing how the wedges are aligned in
/// this Domain, see the documentation for Shell.
class Sphere : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular>>,
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::Wedge<3>,
          CoordinateMaps::EquatorialCompression,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<2>>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::CubicScale<3>,
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
              domain::CoordinateMaps::TimeDependent::Rotation<2>,
              domain::CoordinateMaps::Identity<1>>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the sphere circumscribing the inner cube."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the Sphere."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,angular]."};
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
        "Options for the boundary conditions.";
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

  struct ExpansionMap {
    static constexpr Options::String help = {
        "Options for a time-dependent CubicScale map"};
    using group = TimeDependentMaps;
  };
  /// \brief The outer boundary or pivot point of the
  /// `domain::CoordinateMaps::TimeDependent::CubicScale` map
  struct ExpansionMapOuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "Outer boundary or pivot point of the map"};
    using group = ExpansionMap;
    static std::string name() noexcept { return "OuterBoundary"; }
  };
  /// \brief The initial value of the rotation angle.
  struct InitialExpansion {
    using type = double;
    static constexpr Options::String help = {
        "Expansion value at initial time."};
    using group = ExpansionMap;
  };
  /// \brief The angular velocity of the rotation.
  struct InitialExpansionVelocity {
    using type = double;
    static constexpr Options::String help = {"The rate of expansion."};
    using group = ExpansionMap;
  };
  /// \brief The radial velocity of the outer boundary.
  struct AsymptoticVelocityOuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic velocity of the outer boundary."};
    using group = ExpansionMap;
  };
  /// \brief The timescale for how fast the outer boundary velocity approaches
  /// its asymptotic value.
  struct DecayTimescaleOuterBoundaryVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The timescale for how fast the outer boundary velocity approaches "
        "its asymptotic value."};
    using group = ExpansionMap;
  };
  /// \brief The name of the function of time to be added to the added to the
  /// DataBox for the rotation-about-the-z-axis map.
  struct ExpansionFunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {"Name of the function of time."};
    using group = ExpansionMap;
    static std::string name() noexcept { return "FunctionOfTimeName"; }
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
      tmpl::list<InitialTime, InitialExpirationDeltaT,
                 ExpansionMapOuterBoundary, InitialExpansion,
                 InitialExpansionVelocity, AsymptoticVelocityOuterBoundary,
                 DecayTimescaleOuterBoundaryVelocity,
                 ExpansionFunctionOfTimeName, InitialRotationAngle,
                 InitialAngularVelocity, RotationAboutZAxisFunctionOfTimeName>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      Sphere_detail::enable_time_dependent_maps_v<Metavariables>,
      tmpl::append<time_dependent_options,
                   time_independent_options<Metavariables>>,
      time_independent_options<Metavariables>>;

  static constexpr Options::String help{
      "Creates a 3D Sphere with seven Blocks.\n"
      "Only one refinement level for all dimensions is currently supported.\n"
      "The number of gridpoints in the radial direction can be set\n"
      "independently of the number of gridpoints in the angular directions.\n"
      "The number of gridpoints along the dimensions of the cube is equal\n"
      "to the number of gridpoints along the angular dimensions of the "
      "wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "directions, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  // Constructor for time-independent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == false or
  // when the metavariables do not define
  // Metavariables::domain::enable_time_dependent_maps)
  Sphere(typename InnerRadius::type inner_radius,
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
  Sphere(double initial_time, std::optional<double> initial_expiration_delta_t,
         double expansion_map_outer_boundary, double initial_expansion,
         double initial_expansion_velocity,
         double asymptotic_velocity_outer_boundary,
         double decay_timescale_outer_boundary_velocity,
         std::string expansion_function_of_time_name,
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

  Sphere() = default;
  Sphere(const Sphere&) = delete;
  Sphere(Sphere&&) noexcept = default;
  Sphere& operator=(const Sphere&) = delete;
  Sphere& operator=(Sphere&&) noexcept = default;
  ~Sphere() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
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
  typename UseEquiangularMap::type use_equiangular_map_ = false;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;

  // Variables for FunctionsOfTime options
  bool enable_time_dependence_;
  double initial_time_;
  std::optional<double> initial_expiration_delta_t_;
  double expansion_map_outer_boundary_;
  double initial_expansion_;
  double initial_expansion_velocity_;
  double asymptotic_velocity_outer_boundary_;
  double decay_timescale_outer_boundary_velocity_;
  std::string expansion_function_of_time_name_;
  double initial_rotation_angle_;
  double initial_angular_velocity_;
  std::string rotation_about_z_axis_function_of_time_name_;
};
}  // namespace creators
}  // namespace domain
