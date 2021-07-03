// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Shell.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct Grid;
struct Logical;
}  // namespace Frame

namespace domain::creators {
void Shell::check_for_parse_errors(const Options::Context& context) const {
  if ((inner_boundary_condition_ != nullptr and
       outer_boundary_condition_ == nullptr) or
      (inner_boundary_condition_ == nullptr and
       outer_boundary_condition_ != nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  if (inner_boundary_condition_ != nullptr and
      which_wedges_ != ShellWedges::All) {
    PARSE_ERROR(context,
                "Can only apply boundary conditions when using the full shell. "
                "Additional cases can be supported by adding them to the Shell "
                "domain creator's create_domain function.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(inner_boundary_condition_) or
      is_none(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a shell");
  }
  if (not radial_partitioning_.empty()) {
    if (not std::is_sorted(radial_partitioning_.begin(),
                           radial_partitioning_.end())) {
      PARSE_ERROR(context,
                  "Specify radial partitioning in ascending order. Specified "
                  "radial partitioning is: "
                      << get_output(radial_partitioning_));
    }
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(
          context,
          "First radial partition must be larger than inner radius, but is: "
              << inner_radius_);
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(
          context,
          "Last radial partition must be smaller than outer radius, but is: "
              << outer_radius_);
    }
    const auto duplicate = std::adjacent_find(radial_partitioning_.begin(),
                                              radial_partitioning_.end());
    if (duplicate != radial_partitioning_.end()) {
      PARSE_ERROR(context, "Radial partitioning contains duplicate element: "
                               << *duplicate);
    }
  }
  if (radial_distribution_.size() != number_of_layers_) {
    PARSE_ERROR(context,
                "Specify a 'RadialDistribution' for every spherical shell. You "
                "specified "
                    << radial_distribution_.size()
                    << " items, but the domain has " << number_of_layers_
                    << " shells.");
  }

  // Ensure that the number of shells included in the compression map is
  // at least one fewer than the number of shells, so that the outermost
  // shell remains unaffected by the compression map (to avoid introducing
  // motion of the outer boundary that might complicate the use of a
  // constraint preserving boundary condition)
  if (number_of_compression_layers_ > number_of_layers_ - 1) {
    PARSE_ERROR(context,
                "The domain has "
                    << number_of_layers_
                    << "shells, and the spherical compression can affect at "
                       "most all but the outermost shell. Therefore, "
                       "NumberOfCompressionLayers can be at most "
                    << number_of_layers_ - 1
                    << ", but NumberOfCompressionLayers is: "
                    << number_of_compression_layers_);
  }
}

// Time-independent constructor
Shell::Shell(
    double inner_radius, double outer_radius, size_t initial_refinement,
    std::array<size_t, 2> initial_number_of_grid_points,
    bool use_equiangular_map, double aspect_ratio,
    std::vector<double> radial_partitioning,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    ShellWedges which_wedges,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      initial_refinement_(initial_refinement),
      initial_number_of_grid_points_(initial_number_of_grid_points),
      use_equiangular_map_(use_equiangular_map),
      aspect_ratio_(aspect_ratio),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_(std::move(radial_distribution)),
      which_wedges_(which_wedges),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
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
      number_of_compression_layers_(0),
      initial_size_map_value_(std::numeric_limits<double>::signaling_NaN()),
      initial_size_map_velocity_(std::numeric_limits<double>::signaling_NaN()),
      initial_size_map_acceleration_(
          std::numeric_limits<double>::signaling_NaN()),
      size_map_function_of_time_name_({}) {
  number_of_layers_ = radial_partitioning_.size() + 1;
  check_for_parse_errors(context);
}

// Time-dependent constructor, with additional options for specifying
// the time-dependent maps
Shell::Shell(
    double initial_time, std::optional<double> initial_expiration_delta_t,
    double expansion_map_outer_boundary, double initial_expansion,
    double initial_expansion_velocity,
    double asymptotic_velocity_outer_boundary,
    double decay_timescale_outer_boundary_velocity,
    std::string expansion_function_of_time_name,
    size_t number_of_compression_layers, double initial_size_map_value,
    double initial_size_map_velocity, double initial_size_map_acceleration,
    std::string size_map_function_of_time_name, double inner_radius,
    double outer_radius, size_t initial_refinement,
    std::array<size_t, 2> initial_number_of_grid_points,
    bool use_equiangular_map, double aspect_ratio,
    std::vector<double> radial_partitioning,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    ShellWedges which_wedges,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      initial_refinement_(initial_refinement),
      initial_number_of_grid_points_(initial_number_of_grid_points),
      use_equiangular_map_(use_equiangular_map),
      aspect_ratio_(aspect_ratio),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_(std::move(radial_distribution)),
      which_wedges_(which_wedges),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
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
      number_of_compression_layers_(number_of_compression_layers),
      initial_size_map_value_(initial_size_map_value),
      initial_size_map_velocity_(initial_size_map_velocity),
      initial_size_map_acceleration_(initial_size_map_acceleration),
      size_map_function_of_time_name_(size_map_function_of_time_name) {
  number_of_layers_ = radial_partitioning_.size() + 1;
  check_for_parse_errors(context);
}

Domain<3> Shell::create_domain() const noexcept {
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      coord_maps = sph_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, 1.0, 1.0, use_equiangular_map_, 0.0,
          false, aspect_ratio_, radial_partitioning_, radial_distribution_,
          which_wedges_);

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};

  if (inner_boundary_condition_ != nullptr) {
    // This assumes 6 wedges making up the shell. If you need to support the
    // FourOnEquator or OneAlongMinusX configurations the below code needs to be
    // updated. This would require adding more boundary condition options to the
    // domain creator.
    const size_t blocks_per_layer =
        which_wedges_ == ShellWedges::All             ? 6
        : which_wedges_ == ShellWedges::FourOnEquator ? 4
                                                      : 1;

    boundary_conditions_all_blocks.resize(blocks_per_layer * number_of_layers_);
    for (size_t block_id = 0; block_id < blocks_per_layer; ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      boundary_conditions_all_blocks[boundary_conditions_all_blocks.size() -
                                     block_id - 1][Direction<3>::upper_zeta()] =
          outer_boundary_condition_->get_clone();
    }
  }

  Domain<3> domain{
      std::move(coord_maps),
      corners_for_radially_layered_domains(
          number_of_layers_, false, {{1, 2, 3, 4, 5, 6, 7, 8}}, which_wedges_),
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

    using CompressionMap =
        domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;
    using CompressionMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap>;

    using CompressionAndCubicScaleMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap,
                              CubicScaleMap>;

    size_t blocks_per_layer = 6;
    if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
      blocks_per_layer = 4;
    } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
      blocks_per_layer = 1;
    }
    const std::vector<std::array<size_t, 3>>::size_type number_of_blocks =
        blocks_per_layer * number_of_layers_;

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps{number_of_blocks};

    // The outermost shell is never deformed by the size map.
    block_maps[number_of_blocks - 1] =
        std::make_unique<CubicScaleMapForComposition>(
            CubicScaleMapForComposition{CubicScaleMap{
                expansion_map_outer_boundary_, expansion_function_of_time_name_,
                expansion_function_of_time_name_ + "OuterBoundary"s}});

    // If the number of shells included in the spherical compression is
    // greater than zero, some blocks will instead have a block map that
    // is a composition of a compression map and an expansion map
    if (number_of_compression_layers_ > 0) {
      block_maps[0] =
          std::make_unique<CompressionAndCubicScaleMapForComposition>(
              domain::push_back(
                  CompressionMapForComposition{CompressionMap{
                      size_map_function_of_time_name_,
                      inner_radius_,
                      radial_partitioning_.at(number_of_compression_layers_ -
                                              1),
                      {{0.0, 0.0, 0.0}}}},
                  CubicScaleMapForComposition{CubicScaleMap{
                      expansion_map_outer_boundary_,
                      expansion_function_of_time_name_,
                      expansion_function_of_time_name_ + "OuterBoundary"s}}));
    } else {
      block_maps[0] = block_maps[number_of_blocks - 1]->get_clone();
    }

    // Fill in the rest of the block maps by cloning the relevant maps
    for (size_t block = 1; block < number_of_blocks - 1; ++block) {
      if (block < blocks_per_layer * number_of_compression_layers_) {
        block_maps[block] = block_maps[0]->get_clone();
      } else {
        block_maps[block] = block_maps[number_of_blocks - 1]->get_clone();
      }
    }

    // Finally, inject the time dependent maps into the corresponding blocks
    for (size_t block = 0; block < number_of_blocks; ++block) {
      domain.inject_time_dependent_map_for_block(block,
                                                 std::move(block_maps[block]));
    }
  }

  return domain;
}

std::vector<std::array<size_t, 3>> Shell::initial_extents() const noexcept {
  std::vector<std::array<size_t, 3>>::size_type num_wedges =
      6 * number_of_layers_;
  if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
    num_wedges = 4 * number_of_layers_;
  } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
    num_wedges = number_of_layers_;
  }
  return {
      num_wedges,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}

std::vector<std::array<size_t, 3>> Shell::initial_refinement_levels()
    const noexcept {
  std::vector<std::array<size_t, 3>>::size_type num_wedges =
      6 * number_of_layers_;
  if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
    num_wedges = 4 * number_of_layers_;
  } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
    num_wedges = number_of_layers_;
  }
  return {num_wedges, make_array<3>(initial_refinement_)};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Shell::functions_of_time() const noexcept {
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

  // CompressionMap FunctionOfTime
  result[size_map_function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_size_map_value_},
                                     {initial_size_map_velocity_},
                                     {initial_size_map_acceleration_},
                                     {0.0}}},
          initial_expiration_time);

  return result;
}
}  // namespace domain::creators
