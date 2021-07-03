// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class EquatorialCompression;
template <size_t VolumeDim>
class Identity;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <size_t Dim>
class Wedge;
namespace TimeDependent {
template <bool InteriorMap>
class SphericalCompression;
template <size_t VolumeDim>
class Rotation;
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace TimeDependent
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Shell_detail {
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
}  // namespace Shell_detail
/// \endcond

namespace domain::creators {
/*!
 * \brief Creates a 3D Domain in the shape of a hollow spherical shell
 * consisting of six wedges.
 *
 * \image html WedgeOrientations.png "The orientation of each wedge in Shell."
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
 * controlling the map. The time-dependent map itself consists of a
 * Rotation map about the z axis except in the first `NumberOfLayers`
 * layers, which instead are a composition of a SphericalCompression size map
 * and a Rotation map about the z axis.
 */
class Shell : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial, CoordinateMaps::Wedge<3>,
          CoordinateMaps::EquatorialCompression,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<2>>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::SphericalCompression<false>,
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
              domain::CoordinateMaps::TimeDependent::Rotation<2>,
              domain::CoordinateMaps::Identity<1>>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {"Inner radius of the Shell."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Outer radius of the Shell."};
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

  struct AspectRatio {
    using type = double;
    static constexpr Options::String help = {
        "The equatorial compression factor."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the shell "
        "between InnerRadius and OuterRadius. They must be given in ascending "
        "order. This should be used if boundaries need to be set at specific "
        "radii. If the number but not the specific locations of the boundaries "
        "are important, use InitialRefinement instead."};
  };

  struct RadialDistribution {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each spherical "
        "shell. The possible values are `Linear` and `Logarithmic`. There must "
        "be N+1 radial distributions specified for N radial partitions."};
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  struct WhichWedges {
    using type = ShellWedges;
    static constexpr Options::String help = {
        "Which wedges to include in the shell."};
    static constexpr type suggested_value() noexcept {
      return ShellWedges::All;
    }
  };

  struct BoundaryConditions {
    static constexpr Options::String help = "The boundary conditions to apply.";
  };

  template <typename BoundaryConditionsBase>
  struct InnerBoundaryCondition {
    static std::string name() noexcept { return "InnerBoundary"; }
    static constexpr Options::String help =
        "Options for the inner boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
    using group = BoundaryConditions;
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static std::string name() noexcept { return "OuterBoundary"; }
    static constexpr Options::String help =
        "Options for the outer boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
    using group = BoundaryConditions;
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

  struct SizeMap {
    static constexpr Options::String help = {
        "Options for a time-dependent size maps."};
    using group = TimeDependentMaps;
  };
  struct NumberOfCompressionLayers {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of radial layers affected by the spherical compression map."};
    using group = SizeMap;
  };
  struct InitialSizeMapValue {
    using type = double;
    static constexpr Options::String help = {"SizeMap value at initial time."};
    using group = SizeMap;
    static std::string name() noexcept { return "InitialValue"; }
  };
  struct InitialSizeMapVelocity {
    using type = double;
    static constexpr Options::String help = {
        "SizeMap velocity at initial time."};
    using group = SizeMap;
    static std::string name() noexcept { return "InitialVelocity"; }
  };
  struct InitialSizeMapAcceleration {
    using type = double;
    static constexpr Options::String help = {
        "SizeMap acceleration at initial time."};
    using group = SizeMap;
    static std::string name() noexcept { return "InitialAcceleration"; }
  };
  struct SizeMapFunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {
        "Names of SizeMap function of time."};
    using group = SizeMap;
    static std::string name() noexcept { return "FunctionOfTimeName"; }
  };

  using basic_options =
      tmpl::list<InnerRadius, OuterRadius, InitialRefinement, InitialGridPoints,
                 UseEquiangularMap, AspectRatio, RadialPartitioning,
                 RadialDistribution, WhichWedges>;

  template <typename Metavariables>
  using time_independent_options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          InnerBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  using time_dependent_options =
      tmpl::list<InitialTime, InitialExpirationDeltaT, InitialRotationAngle,
                 InitialAngularVelocity, RotationAboutZAxisFunctionOfTimeName,
                 NumberOfCompressionLayers, InitialSizeMapValue,
                 InitialSizeMapVelocity, InitialSizeMapAcceleration,
                 SizeMapFunctionOfTimeName>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      Shell_detail::enable_time_dependent_maps_v<Metavariables>,
      tmpl::append<time_dependent_options,
                   time_independent_options<Metavariables>>,
      time_independent_options<Metavariables>>;

  static constexpr Options::String help{
      "Creates a 3D spherical shell with 6 Blocks. `UseEquiangularMap` has\n"
      "a default value of `true` because there is no central Block in this\n"
      "domain. Equidistant coordinates are best suited to Blocks with\n"
      "Cartesian grids. However, the option is allowed for testing "
      "purposes. The `aspect_ratio` moves grid points on the shell towards\n"
      "the equator for values greater than 1.0, and towards the poles for\n"
      "positive values less than 1.0. The user may also choose to use only a "
      "single wedge (along the -x direction), or four wedges along the x-y "
      "plane using the `WhichWedges` option. Using the RadialPartitioning "
      "option, a user may set the locations of boundaries of radial "
      "partitions, each of which will have the grid points and refinement "
      "specified from the previous options. The RadialDistribution option "
      "specifies whether the radial grid points are distributed linearly or "
      "logarithmically for each radial partition. Therefore, there must be N+1 "
      "radial distributions specified for N radial partitions. For simple "
      "h-refinement where the number but not the locations of the radial "
      "boundaries are important, the InitialRefinement option should be used "
      "instead of RadialPartitioning. Note that the domain optionally includes "
      "time-dependent maps; enabling "
      "the time-dependent maps requires adding a "
      "struct named domain to the Metavariables, with this "
      "struct conforming to domain::protocols::Metavariables. To enable the "
      "time-dependent maps, set "
      "Metavariables::domain::enable_time_dependent_maps to "
      "true."};

  // Constructor for time-independent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == false or
  // when the metavariables do not define
  // Metavariables::domain::enable_time_dependent_maps)
  Shell(double inner_radius, double outer_radius, size_t initial_refinement,
        std::array<size_t, 2> initial_number_of_grid_points,
        bool use_equiangular_map = true, double aspect_ratio = 1.0,
        std::vector<double> radial_partitioning = {},
        std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
            {domain::CoordinateMaps::Distribution::Linear},
        ShellWedges = ShellWedges::All,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            inner_boundary_condition = nullptr,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            outer_boundary_condition = nullptr,
        const Options::Context& context = {});

  // Constructor for time-dependent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == true),
  // with parameters corresponding to the additional options
  Shell(double initial_time, std::optional<double> initial_expiration_delta_t,
        double initial_rotation_angle, double initial_angular_velocity,
        std::string rotation_about_z_axis_function_of_time_name,
        size_t number_of_compression_layers, double initial_size_map_value,
        double initial_size_map_velocity, double initial_size_map_acceleration,
        std::string size_map_function_of_time_name, double inner_radius,
        double outer_radius, size_t initial_refinement,
        std::array<size_t, 2> initial_number_of_grid_points,
        bool use_equiangular_map = true, double aspect_ratio = 1.0,
        std::vector<double> radial_partitioning = {},
        std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
            {domain::CoordinateMaps::Distribution::Linear},
        ShellWedges = ShellWedges::All,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            inner_boundary_condition = nullptr,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            outer_boundary_condition = nullptr,
        const Options::Context& context = {});

  Shell() = default;
  Shell(const Shell&) = delete;
  Shell(Shell&&) noexcept = default;
  Shell& operator=(const Shell&) = delete;
  Shell& operator=(Shell&&) noexcept = default;
  ~Shell() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  void check_for_parse_errors(const Options::Context& context) const;

  double inner_radius_{};
  double outer_radius_{};
  size_t initial_refinement_{};
  std::array<size_t, 2> initial_number_of_grid_points_{};
  bool use_equiangular_map_ = true;
  double aspect_ratio_ = 1.0;
  std::vector<double> radial_partitioning_ = {};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  ShellWedges which_wedges_ = ShellWedges::All;
  size_t number_of_layers_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;

  // Variables for FunctionsOfTime options
  bool enable_time_dependence_;
  double initial_time_;
  std::optional<double> initial_expiration_delta_t_;
  double initial_rotation_angle_;
  double initial_angular_velocity_;
  std::string rotation_about_z_axis_function_of_time_name_;
  size_t number_of_compression_layers_;
  double initial_size_map_value_;
  double initial_size_map_velocity_;
  double initial_size_map_acceleration_;
  std::string size_map_function_of_time_name_;
};
}  // namespace domain::creators
