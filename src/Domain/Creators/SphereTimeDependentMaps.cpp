// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/SphereTimeDependentMaps.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::creators::sphere {
TimeDependentMapOptions::TimeDependentMapOptions(
    const double initial_time, const ShapeMapOptions& shape_map_options,
    std::optional<ExpansionOptions> expansion_map_options,
    std::optional<RotationOptions> rotation_options)
    : initial_time_(initial_time),
      initial_l_max_(shape_map_options.l_max),
      initial_shape_values_(shape_map_options.initial_values),
      expansion_map_options_(expansion_map_options),
      rotation_options_(rotation_options) {}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions::create_functions_of_time(
    const double inner_radius,
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {size_name, std::numeric_limits<double>::infinity()},
      {shape_name, std::numeric_limits<double>::infinity()},
      {expansion_name, std::numeric_limits<double>::infinity()},
      {expansion_outer_boundary_name, std::numeric_limits<double>::infinity()},
      {rotation_name, std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  DataVector shape_zeros{
      ylm::Spherepack::spectral_size(initial_l_max_, initial_l_max_), 0.0};
  DataVector shape_func{};
  DataVector size_func{1, 0.0};

  if (initial_shape_values_.has_value()) {
    if (std::holds_alternative<KerrSchildFromBoyerLindquist>(
            initial_shape_values_.value())) {
      const ylm::Spherepack ylm{initial_l_max_, initial_l_max_};
      const auto& mass_and_spin =
          std::get<KerrSchildFromBoyerLindquist>(initial_shape_values_.value());
      const DataVector radial_distortion =
          1.0 - get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
                    inner_radius, ylm.theta_phi_points(), mass_and_spin.mass,
                    mass_and_spin.spin)) /
                    inner_radius;
      shape_func = ylm.phys_to_spec(radial_distortion);
      // Transform from SPHEREPACK to actual Ylm for size func
      size_func[0] = shape_func[0] * sqrt(0.5 * M_PI);
      // Set l=0 for shape map to 0 because size is going to be used
      shape_func[0] = 0.0;
    }
  } else {
    shape_func = shape_zeros;
    size_func[0] = 0.0;
  }

  // ShapeMap FunctionOfTime
  result[shape_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {std::move(shape_func), shape_zeros, shape_zeros}},
          expiration_times.at(shape_name));

  DataVector size_deriv{1, 0.0};
  DataVector size_2nd_deriv{1, 0.0};

  // Size FunctionOfTime (used in ShapeMap)
  result[size_name] = std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
      initial_time_,
      std::array<DataVector, 4>{{std::move(size_func),
                                 std::move(size_deriv),
                                 std::move(size_2nd_deriv),
                                 {0.0}}},
      expiration_times.at(size_name));

  // Expansion function of time
  if (expansion_map_options_.has_value()) {
    DataVector expansion_f{
        1, std::get<0>(expansion_map_options_.value().initial_values)};
    DataVector expansion_df{
        1, std::get<1>(expansion_map_options_.value().initial_values)};
    DataVector expansion_d2f{
        1, std::get<2>(expansion_map_options_.value().initial_values)};
    const std::array<DataVector, 3> initial_func_and_derivs{
        expansion_f, expansion_df, expansion_d2f};
    const double t_match = expansion_map_options_.value().match_time;
    const double t_decay = expansion_map_options_.value().decay_time;
    result[expansion_name] =
        std::make_unique<FunctionsOfTime::SettleToConstant>(
            std::move(initial_func_and_derivs), std::move(t_match),
            std::move(t_decay));
  }

  // Expansion outer boundary function of time
  if (expansion_map_options_.has_value()) {
    DataVector expansion_f{
        1, std::get<0>(
               expansion_map_options_.value().initial_values_outer_boundary)};
    DataVector expansion_df{
        1, std::get<1>(
               expansion_map_options_.value().initial_values_outer_boundary)};
    DataVector expansion_d2f{
        1, std::get<2>(
               expansion_map_options_.value().initial_values_outer_boundary)};
    const std::array<DataVector, 3> initial_func_and_derivs{
        expansion_f, expansion_df, expansion_d2f};
    const double t_match = expansion_map_options_.value().match_time;
    const double t_decay = expansion_map_options_.value().decay_time;
    result[expansion_outer_boundary_name] =
        std::make_unique<FunctionsOfTime::SettleToConstant>(
            std::move(initial_func_and_derivs), std::move(t_match),
            std::move(t_decay));
  }

  // Rotation function of time
  if (rotation_options_.has_value()) {
    const DataVector f{
        get<0>(rotation_options_.value().initial_quaternion_values),
        get<1>(rotation_options_.value().initial_quaternion_values),
        get<2>(rotation_options_.value().initial_quaternion_values),
        get<3>(rotation_options_.value().initial_quaternion_values)};
    const DataVector df{
        get<0>(rotation_options_.value().initial_quaternion_values),
        get<1>(rotation_options_.value().initial_quaternion_values),
        get<2>(rotation_options_.value().initial_quaternion_values),
        get<3>(rotation_options_.value().initial_quaternion_values)};
    const DataVector d2f{
        get<0>(rotation_options_.value().initial_quaternion_second_derivatives),
        get<1>(rotation_options_.value().initial_quaternion_second_derivatives),
        get<2>(rotation_options_.value().initial_quaternion_second_derivatives),
        get<3>(
            rotation_options_.value().initial_quaternion_second_derivatives)};
    const double t_match = expansion_map_options_.value().match_time;
    const double t_decay = expansion_map_options_.value().decay_time;
    const std::array<DataVector, 3> initial_func_and_derivs{f, df, d2f};
    result[rotation_name] = std::make_unique<FunctionsOfTime::SettleToConstant>(
        std::move(initial_func_and_derivs), std::move(t_match),
        std::move(t_decay));
  }

  return result;
}

void TimeDependentMapOptions::build_maps(const std::array<double, 3>& center,
                                         const double inner_radius,
                                         const double outer_radius) {
  std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                      ShapeMapTransitionFunction>
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(inner_radius, outer_radius);
  shape_map_ = ShapeMap{center,         initial_l_max_,
                        initial_l_max_, std::move(transition_func),
                        shape_name,     size_name};
  if (expansion_map_options_.has_value()) {
    const double outer_boundary_radius =
        expansion_map_options_.value().outer_boundary_radius;
    const std::string exp_name{expansion_name};
    const std::string exp_outer_name{expansion_outer_boundary_name};
    (*expansion_map_) =
        ExpansionMap{outer_boundary_radius, exp_name, exp_outer_name};
  }
  if (rotation_options_.has_value()) {
    (*rotation_map_) = RotationMap{rotation_name};
  }
}

// If you edit any of the functions below, be sure to update the documentation
// in the Sphere domain creator as well as this class' documentation.
TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    if (expansion_map_.has_value() and rotation_map_.has_value()) {
      return std::make_unique<detail::di_map<ExpansionMap, RotationMap>>(
          expansion_map_.value(), rotation_map_.value());
    } else if (expansion_map_.has_value()) {
      return std::make_unique<detail::di_map<ExpansionMap>>(
          expansion_map_.value());
    } else if (rotation_map_.has_value()) {
      return std::make_unique<detail::di_map<RotationMap>>(
          rotation_map_.value());
    } else {
      return std::make_unique<
          IdentityForComposition<Frame::Distorted, Frame::Inertial>>(
          IdentityMap{});
    }
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    return std::make_unique<detail::gd_map<ShapeMap>>(shape_map_);
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    if (expansion_map_.has_value() and rotation_map_.has_value()) {
      return std::make_unique<
          detail::gi_map<ShapeMap, ExpansionMap, RotationMap>>(
          shape_map_, expansion_map_.value(), rotation_map_.value());
    } else if (expansion_map_.has_value()) {
      return std::make_unique<detail::gi_map<ShapeMap, ExpansionMap>>(
          shape_map_, expansion_map_.value());
    } else if (rotation_map_.has_value()) {
      return std::make_unique<detail::gi_map<ShapeMap, RotationMap>>(
          shape_map_, rotation_map_.value());
    } else {
      return std::make_unique<detail::gi_map<ShapeMap>>(shape_map_);
    }
  } else {
    if (expansion_map_.has_value() and rotation_map_.has_value()) {
      return std::make_unique<detail::gi_map<ExpansionMap, RotationMap>>(
          expansion_map_.value(), rotation_map_.value());
    } else if (expansion_map_.has_value()) {
      return std::make_unique<detail::gi_map<ExpansionMap>>(
          expansion_map_.value());
    } else if (rotation_map_.has_value()) {
      return std::make_unique<detail::gi_map<RotationMap>>(
          rotation_map_.value());
    } else {
      return std::make_unique<
          IdentityForComposition<Frame::Grid, Frame::Inertial>>(IdentityMap{});
    }
  }
}
}  // namespace domain::creators::sphere
