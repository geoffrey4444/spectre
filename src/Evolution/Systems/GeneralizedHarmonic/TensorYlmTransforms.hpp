// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace ylm {
class Spherepack;
}  // namespace ylm
/// \endcond

namespace ylm::TensorYlm {

/*!
 * \brief Transform Generalized Harmonic variables on a spherical shell to
 * tensor-Ylm coefficients of their spatial pieces.
 *
 * The `gh_vars` are nodal values on either a spherical slice
 * (`radial_extents == 1`) or a shell with `radial_extents` radial points and
 * S2 angular collocation points described by `spherepack`. The output
 * `gh_spatial_tensor_ylm_coefficients` has the GH spacetime tensors broken
 * into spatial pieces, transformed to the grid frame using
 * `jac_inertial_to_grid`, converted from nodal to modal S2 coefficients, and
 * finally transformed from Cartesian components to the TensorYlm
 * \f$(\ell, m, \bar m)\f$ basis.
 *
 * The output uses Spherepack spectral storage with the radial offsets
 * interleaved in the same layout as `Spherepack::phys_to_spec_all_offsets`.
 * Its tensor component labels are not Cartesian component labels. They are
 * TensorYlm basis labels: component index 0 denotes an \f$\ell\f$ basis index,
 * component index 1 denotes an \f$m\f$ basis index, and component index 2
 * denotes an \f$\bar m\f$ basis index.
 *
 * For performance, the function does not allocate large buffers and does not
 * build the cartesian-to-spherical matrices. The caller supplies
 * `temp_storage` and precomputed matrices. `temp_storage` is used for
 * grid-frame nodal spatial pieces and for one spectral tensor at a time, so it
 * must have enough storage for the physical GH spatial variables and for the
 * largest spectral GH spatial tensor.
 */
void gh_variables_to_tensor_ylm_coefficients(
    gsl::not_null<Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        gh_spatial_tensor_ylm_coefficients,
    gsl::not_null<Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        temp_storage,
    const Variables<filter_detail::gh_spacetime_vars_list>& gh_vars,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const SimpleSparseMatrix& cart_to_sphere_matrix_i,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ii,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ij,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ijj,
    const Spherepack& spherepack, size_t radial_extents);

}  // namespace ylm::TensorYlm
