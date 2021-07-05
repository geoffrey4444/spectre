// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <cmath>
#include <memory>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;  // IWYU pragma: keep
struct BlockLogical;  // IWYU pragma: keep
}  // namespace Frame

namespace domain::creators {
void Sphere::check_for_parse_errors(const Options::Context& context) const {
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a Sphere");
  }
}

// Time-independent constructor
Sphere::Sphere(typename InnerRadius::type inner_radius,
               typename OuterRadius::type outer_radius,
               typename InitialRefinement::type initial_refinement,
               typename InitialGridPoints::type initial_number_of_grid_points,
               typename UseEquiangularMap::type use_equiangular_map,
               std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
                   boundary_condition,
               const Options::Context& context)
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                // NOLINT
      outer_radius_(std::move(outer_radius)),                // NOLINT
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),  // NOLINT
      boundary_condition_(std::move(boundary_condition)),
      enable_time_dependence_(false),
      initial_time_(std::numeric_limits<double>::signaling_NaN()),
      initial_expiration_delta_t_(std::numeric_limits<double>::signaling_NaN()),
      expansion_map_outer_boundary_(
          std::numeric_limits<double>::signaling_NaN()),
      initial_expansion_(std::numeric_limits<double>::signaling_NaN()),
      initial_expansion_velocity_(std::numeric_limits<double>::signaling_NaN()),
      asymptotic_velocity_outer_boundary_(
          std::numeric_limits<double>::signaling_NaN()),
      decay_timescale_outer_boundary_velocity_(
          std::numeric_limits<double>::signaling_NaN()),
      expansion_function_of_time_name_({}),
      initial_rotation_angle_(std::numeric_limits<double>::signaling_NaN()),
      initial_angular_velocity_(std::numeric_limits<double>::signaling_NaN()),
      rotation_about_z_axis_function_of_time_name_({}) {
  check_for_parse_errors(context);
}

// Time-dependent constructor
Sphere::Sphere(double initial_time,
               std::optional<double> initial_expiration_delta_t,
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
                   boundary_condition,
               const Options::Context& context)
    : inner_radius_(std::move(inner_radius)),                // NOLINT
      outer_radius_(std::move(outer_radius)),                // NOLINT
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),  // NOLINT
      boundary_condition_(std::move(boundary_condition)),
      enable_time_dependence_(true),
      initial_time_(initial_time),
      initial_expiration_delta_t_(initial_expiration_delta_t),
      expansion_map_outer_boundary_(expansion_map_outer_boundary),
      initial_expansion_(initial_expansion),
      initial_expansion_velocity_(initial_expansion_velocity),
      asymptotic_velocity_outer_boundary_(asymptotic_velocity_outer_boundary),
      decay_timescale_outer_boundary_velocity_(
          decay_timescale_outer_boundary_velocity),
      expansion_function_of_time_name_(expansion_function_of_time_name),
      initial_rotation_angle_(initial_rotation_angle),
      initial_angular_velocity_(initial_angular_velocity),
      rotation_about_z_axis_function_of_time_name_(
          std::move(rotation_about_z_axis_function_of_time_name)) {
  check_for_parse_errors(context);
}

Domain<3> Sphere::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(1, true);

  std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coord_maps = sph_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, 0.0, 1.0, use_equiangular_map_);
  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Equiangular3D{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0))}));
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    boundary_conditions_all_blocks.resize(7);
    ASSERT(coord_maps.size() == 7,
           "The number of blocks for which coordinate maps and boundary "
           "conditions are specified should be 7 but the coordinate maps is: "
               << coord_maps.size());
    for (size_t block_id = 0;
         block_id < boundary_conditions_all_blocks.size() - 1; ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
          boundary_condition_->get_clone();
    }
  }

  Domain<3> domain{std::move(coord_maps),
                   corners,
                   {},
                   std::move(boundary_conditions_all_blocks)};
  if (enable_time_dependence_) {
    // Note on frames: Because the relevant maps will all be composed before
    // they are used, all maps here go from Frame::Grid (the frame after the
    // final time-independent map is applied) to Frame::Inertial
    // (the frame after the final time-dependent map is applied).
    using CubicScaleMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
    using CubicScaleMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>;

    using IdentityMap1D = domain::CoordinateMaps::Identity<1>;
    using RotationMap2D = domain::CoordinateMaps::TimeDependent::Rotation<2>;
    using RotationMap =
        domain::CoordinateMaps::TimeDependent::ProductOf2Maps<RotationMap2D,
                                                              IdentityMap1D>;
    using RotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotationMap>;

    using CubicScaleAndRotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap,
                              RotationMap>;

    constexpr size_t number_of_blocks{7};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps{number_of_blocks};

    // All blocks get the same time-dependent maps: expansion and rotation
    block_maps[number_of_blocks - 1] =
        std::make_unique<CubicScaleAndRotationMapForComposition>(
            domain::push_back(
                CubicScaleMapForComposition{CubicScaleMap{
                    expansion_map_outer_boundary_,
                    expansion_function_of_time_name_,
                    expansion_function_of_time_name_ + "OuterBoundary"s}},
                RotationMapForComposition{RotationMap{
                    RotationMap2D{rotation_about_z_axis_function_of_time_name_},
                    IdentityMap1D{}}}));

    // Fill in the rest of the block maps by cloning the relevant maps
    for (size_t block = 0; block < number_of_blocks - 1; ++block) {
      block_maps[block] = block_maps[number_of_blocks - 1]->get_clone();
    }

    // Finally, inject the time dependent maps into the corresponding blocks
    for (size_t block = 0; block < number_of_blocks; ++block) {
      domain.inject_time_dependent_map_for_block(block,
                                                 std::move(block_maps[block]));
    }
  }

  return domain;
}

std::vector<std::array<size_t, 3>> Sphere::initial_extents() const noexcept {
  std::vector<std::array<size_t, 3>> extents{
      6,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
  extents.push_back(
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[1]}});
  return extents;
}

std::vector<std::array<size_t, 3>> Sphere::initial_refinement_levels() const
    noexcept {
  return {7, make_array<3>(initial_refinement_)};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Sphere::functions_of_time() const noexcept {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  if (not enable_time_dependence_) {
    return result;
  }

  const double initial_expiration_time =
      initial_expiration_delta_t_ ? initial_time_ + *initial_expiration_delta_t_
                                  : std::numeric_limits<double>::infinity();

  // ExpansionMap FunctionOfTime
  result[expansion_function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {{initial_expansion_}, {initial_expansion_velocity_}, {0.0}}},
          initial_expiration_time);
  result[expansion_function_of_time_name_ + "OuterBoundary"s] =
      std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
          1.0, initial_time_, asymptotic_velocity_outer_boundary_,
          decay_timescale_outer_boundary_velocity_);

  // RotationAboutZAxisMap FunctionOfTime for the rotation angle about the z
  // axis \f$\phi\f$.
  result[rotation_about_z_axis_function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_rotation_angle_},
                                     {initial_angular_velocity_},
                                     {0.0},
                                     {0.0}}},
          initial_expiration_time);

  return result;
}
}  // namespace domain::creators
