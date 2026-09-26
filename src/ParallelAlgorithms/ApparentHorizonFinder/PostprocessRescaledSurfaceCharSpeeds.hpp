// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"

/// \cond
template <size_t Dim>
class Domain;
/// \endcond

namespace ah {
/*!
 * \brief Sample characteristic speeds from complete saved volume data at one
 * time, using the same preparation and interpolation as the evolution.
 *
 * The three element vectors must have the same size. Spacetime metrics are
 * inertial-frame tensors at the corresponding mesh points. Only blocks selected
 * by `blocks` are used; names of block groups are also accepted. By default,
 * all stationary blocks and blocks with a distorted frame are selected.
 *
 * The horizon and its optional time derivative must have the same angular
 * resolution and fixed distorted-frame expansion center as the excision sphere.
 * Missing derivatives produce MissingTimeDerivative, and incomplete saved
 * volume coverage produces MissingBlockCoverage. The returned row follows
 * rescaled_surface_char_speed_legend, including these statuses and NaN speeds.
 * Invalid input sizes, duplicate elements, and missing or expired functions of
 * time throw std::invalid_argument.
 */
std::vector<double> postprocess_rescaled_surface_char_speeds(
    const ylm::Strahlkorper<Frame::Distorted>& horizon,
    const std::optional<ylm::Strahlkorper<Frame::Distorted>>&
        time_deriv_horizon,
    const std::vector<ElementId<3>>& element_ids,
    const std::vector<Mesh<3>>& meshes,
    const std::vector<tnsr::aa<DataVector, 3>>& spacetime_metrics,
    const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, double time,
    const RescaledSurfaceCharSpeedOptions& options,
    const std::optional<std::vector<std::string>>& blocks = std::nullopt);
}  // namespace ah
