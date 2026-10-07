// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/RadialProjectionSourceIncrement.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace gh {
DataVector radial_projection_source_increment(
    const RadialSourceProjection& projection,
    const Variables<typename System<3>::variables_tag::tags_list>& state,
    const Mesh<3>& mesh, const double time,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& logical_coordinates,
    const ElementMap<3, Frame::Grid>& element_to_grid,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coarse_coordinates,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& coarse_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        coarse_mesh_velocity,
    const gauges::GaugeCondition& gauge, const bool preserve_inner) {
  const size_t n = mesh.extents(0);
  const size_t q = projection.evaluation_points();
  if (projection.radial_points() != n or
      mesh.number_of_grid_points() != state.number_of_grid_points() or
      mesh.basis(0) != Spectral::Basis::Legendre or
      mesh.quadrature(0) != Spectral::Quadrature::GaussLobatto or
      mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(2) != Spectral::Basis::SphericalHarmonic or
      mesh.quadrature(1) != Spectral::Quadrature::Gauss or
      mesh.quadrature(2) != Spectral::Quadrature::Equiangular) {
    ERROR(
        "Experimental radial source projection requires a matching spherical "
        "shell mesh with radial Legendre/GaussLobatto points.");
  }
  const size_t angular_points = mesh.extents(1) * mesh.extents(2);
  const size_t high_points = q * angular_points;
  // Avoid high-extent Spectral coordinate caches; only the original angular
  // coordinates are reused, and the bounded operator supplies radial nodes.
  tnsr::I<DataVector, 3, Frame::ElementLogical> high_logical(high_points);
  for (size_t a = 0; a < angular_points; ++a) {
    for (size_t r = 0; r < q; ++r) {
      high_logical.get(0)[r + q * a] = projection.evaluation_nodes()[r];
      high_logical.get(1)[r + q * a] = logical_coordinates.get(1)[n * a];
      high_logical.get(2)[r + q * a] = logical_coordinates.get(2)[n * a];
    }
  }
  const auto high_grid = element_to_grid(high_logical);
  const auto logical_to_grid_inverse =
      element_to_grid.inv_jacobian(high_logical);
  auto mapped = grid_to_inertial.coords_frame_velocity_jacobians(
      high_grid, time, functions_of_time);
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      high_inverse(high_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        high_inverse.get(i, j) +=
            logical_to_grid_inverse.get(i, k) * std::get<1>(mapped).get(k, j);
      }
    }
  }
  constexpr size_t components = 50;
  DataVector state_values(state.size());
  std::copy(state.data(), state.data() + state.size(), state_values.begin());
  const auto prolonged = projection.prolong(state_values, components);
  Variables<typename System<3>::variables_tag::tags_list> high_state(
      high_points);
  std::copy(prolonged.begin(), prolonged.end(), high_state.data());
  const std::array<size_t, 3> high_extents{
      {q, mesh.extents(1), mesh.extents(2)}};
  const Mesh<3> high_mesh(high_extents, mesh.basis(), mesh.quadrature());
  const auto coarse_source = radial_projection_local_source(
      state, mesh, time, coarse_coordinates, coarse_inverse_jacobian,
      coarse_mesh_velocity, gauge);
  const auto high_source = radial_projection_local_source(
      high_state, high_mesh, time, std::get<0>(mapped), high_inverse,
      std::get<3>(mapped), gauge);
  DataVector coarse_values(coarse_source.size()),
      high_values(high_source.size());
  std::copy(coarse_source.data(), coarse_source.data() + coarse_source.size(),
            coarse_values.begin());
  std::copy(high_source.data(), high_source.data() + high_source.size(),
            high_values.begin());
  auto increment = projection.source_increment(coarse_values, high_values,
                                               components, preserve_inner);
  if (not std::all_of(increment.begin(), increment.end(),
                      [](const double x) { return std::isfinite(x); })) {
    ERROR("Nonfinite experimental radial source increment.");
  }
  return increment;
}
}  // namespace gh
