// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Newman Penrose quantity \f$\Psi_4\f$ using the characteristic
 * field U\f$^{8+}\f$ and complex vector \f$\bar{m}^i\f$.
 *
 * \details Computes \f$\Psi_4\f$ as: \f$\Psi_4 =
 * U^{8+}_{ij}\bar{m}^i\bar{m}^j\f$ with the characteristic field
 * \f$U^{8+} = (P^{(a}_i P^{b)}_j - \frac{1}{2}P_{ij}P^{ab})
 * (E_{ab} - \epsilon_a^{cd}n_dB_{cb}\f$)
 * and \f$\bar{m}^i\f$ = \f$(\hat{x}^i-i\hat{y}^i)\f$. The first polarization
 * vector \f$\hat{x}^i\f$ is constructed by projecting the coordinate
 * \f$x\f$ direction orthogonal to the radial direction, with the coordinate
 * \f$y\f$ direction used as a fallback near the \f$x\f$ axis. The second
 * polarization vector \f$\hat{y}^i\f$ is the coordinate \f$y\f$ direction
 * Gram-Schmidt orthonormalized against the radial direction and
 * \f$\hat{x}^i\f$. Where those directions become linearly dependent, the
 * metric cross product of the radial direction and \f$\hat{x}^i\f$ is used as
 * a fallback. At the origin, where the radial direction is undefined, the
 * historical Cartesian \f$x\f$-\f$y\f$ basis is used.
 *
 */
template <typename Frame>
void psi_4(gsl::not_null<Scalar<ComplexDataVector>*> psi_4_result,
           const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
           const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
           const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
           const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
           const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
           const tnsr::I<DataVector, 3, Frame>& inertial_coords);

template <typename Frame>
Scalar<ComplexDataVector> psi_4(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);
/// @}

}  // namespace gr
