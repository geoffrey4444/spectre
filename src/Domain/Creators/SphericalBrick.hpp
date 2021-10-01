// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class SphericalBrick.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {

/// Create a 3D Domain consisting of a single Block.
class SphericalBrick : public DomainCreator<3> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       domain::CoordinateMaps::Wedge<3>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {"Inner radius."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Outer radius."};
  };
  struct SphericityInner {
    using type = double;
    static constexpr Options::String help = {"Inner sphericity."};
  };
  struct SphericityOuter {
    using type = double;
    static constexpr Options::String help = {"Outer sphericity."};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, 3>;
    static constexpr Options::String help = {
        "Sequence for [x,y,z], true if periodic."};
  };

  struct InitialRefinement {
    using type = std::array<size_t, 3>;
    static constexpr Options::String help = {
        "Initial refinement level in [x,y,z]."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 3>;
    static constexpr Options::String help = {
        "Initial number of grid points in [x,y,z]."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() noexcept { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "The boundary condition to impose on all sides except xi.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryConditionUpperXi {
    static std::string name() noexcept { return "BoundaryConditionUpperXi"; }
    static constexpr Options::String help =
        "The boundary condition to impose on upper xi.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using common_options =
      tmpl::list<InnerRadius, OuterRadius, SphericityInner, SphericityOuter,
                 InitialRefinement, InitialGridPoints>;
  using options_periodic = tmpl::list<IsPeriodicIn>;

  template <typename Metavariables>
  using options = tmpl::append<
      common_options,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<
              BoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              BoundaryConditionUpperXi<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>,
          options_periodic>,
      tmpl::list<TimeDependence>>;

  static constexpr Options::String help{"Creates a 3D brick."};

  SphericalBrick(
      double inner_radius, double outer_radius, double sphericity_inner,
      double sphericity_outer,
      typename InitialRefinement::type initial_refinement_level_xyz,
      typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
      typename IsPeriodicIn::type is_periodic_in_xyz,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr) noexcept;

  SphericalBrick(
      double inner_radius, double outer_radius, double sphericity_inner,
      double sphericity_outer,
      typename InitialRefinement::type initial_refinement_level_xyz,
      typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          boundary_condition_upper_xi = nullptr,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      const Options::Context& context = {});

  SphericalBrick() = default;
  SphericalBrick(const SphericalBrick&) = delete;
  SphericalBrick(SphericalBrick&&) noexcept = default;
  SphericalBrick& operator=(const SphericalBrick&) = delete;
  SphericalBrick& operator=(SphericalBrick&&) noexcept = default;
  ~SphericalBrick() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  double inner_radius_, outer_radius_, sphericity_inner_, sphericity_outer_;
  typename IsPeriodicIn::type is_periodic_in_xyz_{};
  typename InitialRefinement::type initial_refinement_level_xyz_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xyz_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_upper_xi_;
};
}  // namespace creators
}  // namespace domain
