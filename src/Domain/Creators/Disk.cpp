// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Disk.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>

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
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

namespace domain::creators {
void Disk::check_for_parse_errors(const Options::Context& context) const {
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (boundary_condition_ != nullptr and is_periodic(boundary_condition_)) {
    PARSE_ERROR(context, "Cannot have periodic boundary conditions on a disk.");
  }
}

Disk::Disk(typename InnerRadius::type inner_radius,
           typename OuterRadius::type outer_radius,
           typename InitialRefinement::type initial_refinement,
           typename InitialGridPoints::type initial_number_of_grid_points,
           typename UseEquiangularMap::type use_equiangular_map,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               boundary_condition,
           const Options::Context& context)
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),         // NOLINT
      outer_radius_(std::move(outer_radius)),         // NOLINT
      initial_refinement_(                            // NOLINT
          std::move(initial_refinement)),             // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      use_equiangular_map_(use_equiangular_map),      // NOLINT
      boundary_condition_(std::move(boundary_condition)),
      enable_time_dependence_(false),
      initial_time_(std::numeric_limits<double>::signaling_NaN()),
      initial_expiration_delta_t_(std::numeric_limits<double>::signaling_NaN()),
      initial_rotation_angle_(std::numeric_limits<double>::signaling_NaN()),
      initial_angular_velocity_(std::numeric_limits<double>::signaling_NaN()),
      rotation_about_z_axis_function_of_time_name_({}) {
  check_for_parse_errors(context);
}

Disk::Disk(double initial_time,
           std::optional<double> initial_expiration_delta_t,
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
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),         // NOLINT
      outer_radius_(std::move(outer_radius)),         // NOLINT
      initial_refinement_(                            // NOLINT
          std::move(initial_refinement)),             // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      use_equiangular_map_(use_equiangular_map),      // NOLINT
      boundary_condition_(std::move(boundary_condition)),
      enable_time_dependence_(true),
      initial_time_(initial_time),
      initial_expiration_delta_t_(initial_expiration_delta_t),
      initial_rotation_angle_(initial_rotation_angle),
      initial_angular_velocity_(initial_angular_velocity),
      rotation_about_z_axis_function_of_time_name_(
          rotation_about_z_axis_function_of_time_name) {
  check_for_parse_errors(context);
}

Domain<2> Disk::create_domain() const noexcept {
  using Wedge2DMap = CoordinateMaps::Wedge<2>;
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;

  std::array<size_t, 4> block0_corners{{1, 5, 3, 7}};  //+x wedge
  std::array<size_t, 4> block1_corners{{3, 7, 2, 6}};  //+y wedge
  std::array<size_t, 4> block2_corners{{2, 6, 0, 4}};  //-x wedge
  std::array<size_t, 4> block3_corners{{0, 4, 1, 5}};  //-y wedge
  std::array<size_t, 4> block4_corners{{0, 1, 2, 3}};  // Center square

  std::vector<std::array<size_t, 4>> corners{block0_corners, block1_corners,
                                             block2_corners, block3_corners,
                                             block4_corners};

  auto coord_maps = make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial>(
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                 use_equiangular_map_},
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                 use_equiangular_map_},
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                 use_equiangular_map_},
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                 use_equiangular_map_});

  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(Equiangular2D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                        inner_radius_ / sqrt(2.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                        inner_radius_ / sqrt(2.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Affine2D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0))}));
  }

  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    for (size_t block_id = 0; block_id < 4; ++block_id) {
      DirectionMap<
          2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
          boundary_conditions{};
      boundary_conditions[Direction<2>::upper_xi()] =
          boundary_condition_->get_clone();
      boundary_conditions_all_blocks.push_back(std::move(boundary_conditions));
    }
    boundary_conditions_all_blocks.emplace_back();
  }

  Domain<2> domain{std::move(coord_maps),
                   corners,
                   {},
                   std::move(boundary_conditions_all_blocks)};
  // Inject the hard-coded time-dependence
  if (enable_time_dependence_) {
    using RotationMap = domain::CoordinateMaps::TimeDependent::Rotation<2>;
    using RotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotationMap>;

    constexpr size_t number_of_blocks{5};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 2>>>
        block_maps{number_of_blocks};
    block_maps[0] =
        std::make_unique<RotationMapForComposition>(RotationMapForComposition{
            RotationMap{rotation_about_z_axis_function_of_time_name_}});

    // Fill in the rest of the block maps by cloning the relevant maps
    for (size_t block = 1; block < number_of_blocks; ++block) {
      block_maps[block] = block_maps[0]->get_clone();
    }

    // Finally, inject the time dependent maps into the corresponding blocks
    for (size_t block = 0; block < number_of_blocks; ++block) {
      domain.inject_time_dependent_map_for_block(block,
                                                 std::move(block_maps[block]));
    }
  }

  return domain;
}

std::vector<std::array<size_t, 2>> Disk::initial_extents() const noexcept {
  return {
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1]}}};
}

std::vector<std::array<size_t, 2>> Disk::initial_refinement_levels()
    const noexcept {
  return {5, make_array<2>(initial_refinement_)};
}

auto Disk::functions_of_time() const noexcept -> std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  if (not enable_time_dependence_) {
    return result;
  }
  const double initial_expiration_time =
      initial_expiration_delta_t_ ? initial_time_ + *initial_expiration_delta_t_
                                  : std::numeric_limits<double>::infinity();

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
