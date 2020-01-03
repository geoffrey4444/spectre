// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Declares function templates to calculate the Ricci tensor

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

/// \endcond

/// \ingroup SpecialRelativityGroup
/// Holds functions related to special relativity.
namespace sr {
// @{
/*!
 * \ingroup SpecialRelativityGroup
 * \brief Computes the matrix for a Lorentz boost from a single
 * velocity vector (i.e., not a velocity field).
 *
 * \details Given a spatial velocity vector \f$v^i\f$ (with \f$c=1\f$),
 * compute the matrix \f$\Lambda^{a}_{\bar{a}}\f$ for a Lorentz boost with
 * that velocity [e.g. Eq. (2.38) of \cite ThorneBlandford2017]:
 *
 * \f{align}{
 * \Lambda^t_{\bar{t}} &= \gamma, \\
 * \Lambda^t_{\bar{i}} = \Lambda^i_{\bar{t}} &= \gamma v^i, \\
 * \Lambda^i_{\bar{j}} = \Lambda^j_{\bar{i}} &= [(\gamma - 1)/v^2] v^i v^j
 *                                              + \delta^{ij}.
 * \f}
 *
 * Here \f$v = \sqrt{\delta_{ij} v^i v^j}\f$, \f$\gamma = 1/\sqrt{1-v^2}\f$,
 * and \f$\delta^{ij}\f$ is the Kronecker delta.
 *
 * Note that while the Lorentz boost matrix has indices with valences as shown
 * above, `Tensor` only supports symmetry for indices of the same valence.
 * Therefore, the matrix is returned as type `tnsr::aa<DataType>`.
 */
template <size_t SpatialDim, typename Frame>
tnsr::aa<double, SpatialDim, Frame> lorentz_boost_matrix(
    const tnsr::I<double, SpatialDim, Frame>& velocity) noexcept;

template <size_t SpatialDim, typename Frame>
void lorentz_boost_matrix(
    gsl::not_null<tnsr::aa<double, SpatialDim, Frame>*> boost_matrix,
    const tnsr::I<double, SpatialDim, Frame>& velocity) noexcept;
// @}
}  // namespace sr
