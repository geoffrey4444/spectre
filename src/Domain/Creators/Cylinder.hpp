// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
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
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
template <size_t VolumeDim>
class Identity;
namespace TimeDependent {
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <size_t VolumeDim>
class Rotation;
}  // namespace TimeDependent
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Cylinder_detail {
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
}  // namespace Cylinder_detail
/// \endcond

namespace domain::creators {
/*!
 *
 * Create a 3D Domain in the shape of a cylinder where the cross-section
 * is a square surrounded by four two-dimensional wedges (see `Wedge`).
 *
 * The outer shell can be split into sub-shells and the cylinder can be split
 * into disks along its height.
 * The block numbering starts at the inner square and goes counter-clockwise,
 * starting with the eastern wedge (+x-direction), through consecutive shells,
 * then repeats this pattern for all layers bottom to top.
 *
 * \image html Cylinder.png "The Cylinder Domain."
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
class Cylinder : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
              domain::CoordinateMaps::TimeDependent::Rotation<2>,
              domain::CoordinateMaps::Identity<1>>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the circle circumscribing the inner square."};
    static double lower_bound() noexcept { return 0.; }
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the cylinder."};
    static double lower_bound() noexcept { return 0.; }
  };

  struct LowerBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the base of the cylinder."};
  };

  struct UpperBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the top of the cylinder."};
  };

  struct IsPeriodicInZ {
    using type = bool;
    static constexpr Options::String help = {
        "True if periodic in the cylindrical z direction."};
  };

  struct InitialRefinement {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level. Specify one of: a single number, a list "
        "representing [r, theta, z], or such a list for every block in the "
        "domain. The central cube always uses the value for 'theta' in both "
        "x- and y-direction."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points. Specify one of: a single number, a "
        "list representing [r, theta, z], or such a list for every block in "
        "the domain. The central cube always uses the value for 'theta' in "
        "both x- and y-direction."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the outer shell "
        "between InnerRadius and OuterRadius."};
  };

  struct HeightPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "z-coordinates of the boundaries splitting the domain into discs "
        "between LowerBound and UpperBound."};
  };

  struct RadialDistribution {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each cylindrical "
        "shell. The innermost shell must have a 'Linear' distribution because "
        "it changes in circularity. The 'RadialPartitioning' determines the "
        "number of shells."};
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  template <typename BoundaryConditionsBase>
  struct LowerBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "Lower"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower base of the "
        "cylinder, i.e. at the `LowerBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct UpperBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "Upper"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the upper base of the "
        "cylinder, i.e. at the `UpperBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct MantleBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "Mantle"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the mantle of the "
        "cylinder, i.e. at the `OuterRadius` in the radial direction.";
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

  template <typename Metavariables>
  using time_independent_options = tmpl::append<
      tmpl::list<InnerRadius, OuterRadius, LowerBound, UpperBound>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<
              LowerBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              UpperBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              MantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>,
          tmpl::list<IsPeriodicInZ>>,
      tmpl::list<InitialRefinement, InitialGridPoints, UseEquiangularMap,
                 RadialPartitioning, HeightPartitioning, RadialDistribution>>;

  using time_dependent_options =
      tmpl::list<InitialTime, InitialExpirationDeltaT, InitialRotationAngle,
                 InitialAngularVelocity, RotationAboutZAxisFunctionOfTimeName>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      Cylinder_detail::enable_time_dependent_maps_v<Metavariables>,
      tmpl::append<time_dependent_options,
                   time_independent_options<Metavariables>>,
      time_independent_options<Metavariables>>;

  static constexpr Options::String help{
      "Creates a right circular Cylinder with a square prism surrounded by \n"
      "wedges. \n"
      "The cylinder can be partitioned radially into multiple cylindrical \n"
      "shells as well as partitioned along the cylinder's height into \n"
      "multiple disks. Including this partitioning, the number of Blocks is \n"
      "given by (1 + 4*(1+n_s)) * (1+n_z), where n_s is the \n"
      "length of RadialPartitioning and n_z the length of \n"
      "HeightPartitioning. The block numbering starts at the inner square \n"
      "and goes counter-clockwise, starting with the eastern wedge \n"
      "(+x-direction) through consecutive shells, then repeats this pattern \n"
      "for all layers bottom to top. The wedges are named as follows: \n"
      "  +x-direction: East \n"
      "  +y-direction: North \n"
      "  -x-direction: West \n"
      "  -y-direction: South \n"
      "The circularity of the wedge changes from 0 to 1 within the first \n"
      "shell.\n"
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

  // Constructors for time-independent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == false or
  // when the metavariables do not define
  // Metavariables::domain::enable_time_dependent_maps)
  Cylinder(
      double inner_radius, double outer_radius, double lower_bound,
      double upper_bound, bool is_periodic_in_z,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map, std::vector<double> radial_partitioning = {},
      std::vector<double> height_partitioning = {},
      std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
          {domain::CoordinateMaps::Distribution::Linear},
      const Options::Context& context = {});

  Cylinder(
      double inner_radius, double outer_radius, double lower_bound,
      double upper_bound,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          lower_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          upper_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          mantle_boundary_condition,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map, std::vector<double> radial_partitioning = {},
      std::vector<double> height_partitioning = {},
      std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
          {domain::CoordinateMaps::Distribution::Linear},
      const Options::Context& context = {});

  // Constructor for time-dependent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == true),
  // with parameters corresponding to the additional options
  Cylinder(
      double initial_time, std::optional<double> initial_expiration_delta_t,
      double initial_rotation_angle, double initial_angular_velocity,
      std::string rotation_about_z_axis_function_of_time_name,
      double inner_radius, double outer_radius, double lower_bound,
      double upper_bound, bool is_periodic_in_z,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map, std::vector<double> radial_partitioning = {},
      std::vector<double> height_partitioning = {},
      std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
          {domain::CoordinateMaps::Distribution::Linear},
      const Options::Context& context = {});

  Cylinder(
      double initial_time, std::optional<double> initial_expiration_delta_t,
      double initial_rotation_angle, double initial_angular_velocity,
      std::string rotation_about_z_axis_function_of_time_name,
      double inner_radius, double outer_radius, double lower_bound,
      double upper_bound,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          lower_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          upper_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          mantle_boundary_condition,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map, std::vector<double> radial_partitioning = {},
      std::vector<double> height_partitioning = {},
      std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
          {domain::CoordinateMaps::Distribution::Linear},
      const Options::Context& context = {});

  Cylinder() = default;
  Cylinder(const Cylinder&) = delete;
  Cylinder(Cylinder&&) noexcept = default;
  Cylinder& operator=(const Cylinder&) = delete;
  Cylinder& operator=(Cylinder&&) noexcept = default;
  ~Cylinder() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  void check_for_parse_errors_and_init(
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      const Options::Context& context);
  void check_bc_parse_errors(const Options::Context& context);

  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  double lower_bound_{std::numeric_limits<double>::signaling_NaN()};
  double upper_bound_{std::numeric_limits<double>::signaling_NaN()};
  bool is_periodic_in_z_{true};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  bool use_equiangular_map_{false};
  std::vector<double> radial_partitioning_{};
  std::vector<double> height_partitioning_{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      upper_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      mantle_boundary_condition_{};

  // Variables for FunctionsOfTime options
  bool enable_time_dependence_;
  double initial_time_;
  std::optional<double> initial_expiration_delta_t_;
  double initial_rotation_angle_;
  double initial_angular_velocity_;
  std::string rotation_about_z_axis_function_of_time_name_;
};
}  // namespace domain::creators
