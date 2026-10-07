// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"

namespace gh {
/// Isolate the native local non-gamma DH source for the experimental radial
/// projection. Gamma1=-1; gamma0=gamma2=0; external spatial derivatives zero.
/// Pi/Phi still supply the native DH spacetime-metric derivative. The caller
/// adds a projected-source difference to the original complete volume RHS,
/// thereby retaining its original principal and gamma-dependent terms.
Variables<typename System<3>::variables_tag::tags_list>
radial_projection_local_source(
    const Variables<typename System<3>::variables_tag::tags_list>& state,
    const Mesh<3>& mesh, double time,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coordinates,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const gauges::GaugeCondition& gauge);
}  // namespace gh
