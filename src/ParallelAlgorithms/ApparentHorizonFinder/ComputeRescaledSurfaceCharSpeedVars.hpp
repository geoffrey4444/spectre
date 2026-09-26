// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Domain;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
/// \endcond

namespace ah {
/// Additional volume fields for rescaled-horizon characteristic speeds.
using rescaled_surface_char_speed_vars =
    tmpl::list<gr::Tags::Lapse<DataVector>,
               gr::Tags::Shift<DataVector, 3, Frame::Distorted>>;

/*!
 * \brief Compute the lapse and the shift in distorted coordinates.
 *
 * The distorted-frame shift includes the velocity of the distorted-to-inertial
 * map, but not the velocity of the grid-to-distorted map:
 *
 * \f{equation}{
 * \beta^{\hat\imath} = \frac{\partial x^{\hat\imath}}{\partial x^j}
 * \left(\beta^j + \left.\partial_t x^j\right|_{x^{\hat\imath}}\right).
 * \f}
 *
 * All coordinates and maps are evaluated at the supplied volume-data time.
 * Stationary domains identify the distorted and inertial frames.
 */
void compute_rescaled_surface_char_speed_vars(
    gsl::not_null<Variables<rescaled_surface_char_speed_vars>*> result,
    const tnsr::aa<DataVector, 3>& spacetime_metric, const Domain<3>& domain,
    const Mesh<3>& mesh, const ElementId<3>& element_id, double time,
    const domain::FunctionsOfTimeMap& functions_of_time);
}  // namespace ah
