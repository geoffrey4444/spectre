// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace domain {
/*!
 * \ingroup ComputationalDomainGroup
 * \brief A diagonstic comparing the analytic and numerical Jacobians for a
 * map.
 *
 * Specifically, returns
 * \f$\sqrt{C_{\hat{i}} C_{\hat{j}}\delta^{\hat{i}\hat{j}}}\f$, where
 * \f[
 * C_{\hat{i}} = 1 -
 * \frac{\sum_i \partial_{\hat{i}} x^i}{\sum_i D_{\hat{i}} x^i + \epsilon}
 * \f], where \f$x^{\hat{i}}\f$ are the logical
 * coordinates, \f$x^i\f$ are the coordinates in the target frame,
 * \f$\partial_{\hat{i}}x^i\f$ is the analytic Jacobian, \f$D_{\hat{i}} x^i\f$
 * is the numerical Jacobian, and \f$\epsilon\f$ is a small value to avoid
 * division by zero.
 *
 * \note If \f$|\partial_{\hat{\imath}}|\f$ and \f$|D_{\hat{\imath}}|\f$ are
 * both less than 1.e-10, then \f$C_{\hat{\imath}}\f$ is set to 0, to avoid
 * dividing a small number by a small number.
 *
 * \note This function accepts the transpose of the numeric Jacobian as a
 * parameter, since the numeric Jacobian will typically be computed via
 * logical_partial_derivative(), which prepends the logical (source frame)
 * derivative index. Tensors of type Jacobian, in contrast, have the derivative
 * index second.
 */
template <size_t Dim, typename Fr>
void jacobian_diagnostic(
    const gsl::not_null<Scalar<DataVector>*> jacobian_diagnostic,
    const Jacobian<DataVector, Dim, typename Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const TensorMetafunctions::prepend_spatial_index<
        tnsr::I<DataVector, Dim, Fr>, Dim, UpLo::Lo,
        typename Frame::ElementLogical>& numeric_jacobian_transpose);

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief A diagonstic comparing the analytic and numerical Jacobians for a
/// map. See `domain::jacobian_diagnostic` for details.
struct JacobianDiagnostic : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Computes the Jacobian diagnostic of the map held by `MapTag` at the
/// coordinates held by `MappedCoordsTag`. The coordinates must be in the target
/// frame of the map. See `domain::jacobian_diagnostic` for details of the
/// calculation.
template <typename MapTag, typename SourceCoordsTag, typename MappedCoordsTag>
struct JacobianDiagnosticCompute : JacobianDiagnostic, db::ComputeTag {
  using base = JacobianDiagnostic;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<MapTag, SourceCoordsTag, MappedCoordsTag, Mesh<MapTag::dim>>;
  static constexpr auto function(
      const gsl::not_null<return_type*> jacobian_diag,
      const typename MapTag::type& element_map,
      const tnsr::I<DataVector, MapTag::dim, typename MapTag::source_frame>&
          source_coords,
      const tnsr::I<DataVector, MapTag::dim, typename MapTag::target_frame>&
          mapped_coords,
      const ::Mesh<MapTag::dim>& mesh) {
    const auto analytic_jacobian =
        determinant_and_inverse(element_map.inv_jacobian(source_coords)).second;
    // Note: Jacobian has the source frame index second, but
    // logical_partial_derivative prepends the logical (source frame) index.
    // So this is actually the transpose of the numerical jacobian.
    const auto numerical_jacobian_transpose =
        logical_partial_derivative(mapped_coords, mesh);

    ::domain::jacobian_diagnostic(jacobian_diag, analytic_jacobian,
                                  numerical_jacobian_transpose);
  }
};
}  // namespace Tags
}  // namespace domain
