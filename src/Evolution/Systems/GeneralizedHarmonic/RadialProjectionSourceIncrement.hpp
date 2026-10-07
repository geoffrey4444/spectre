// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/RadialProjectionLocalSource.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/RadialSourceProjection.hpp"

namespace gh {
/// Compute only R S0(I u) - S0(u), using the actual moving maps. Neither the
/// evolved variables nor the existing volume/face temporary data are modified.
/// The caller retains the original full RHS, including its principal,
/// constraint-damping, mesh-advection and boundary terms.
DataVector radial_projection_source_increment(
    const RadialSourceProjection& projection,
    const Variables<typename System<3>::variables_tag::tags_list>& state,
    const Mesh<3>& mesh, double time,
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
    const gauges::GaugeCondition& gauge, bool preserve_inner);
}  // namespace gh
