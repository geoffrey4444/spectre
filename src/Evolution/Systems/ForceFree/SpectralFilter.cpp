// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
using ff_tags = tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                           ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                           ForceFree::Tags::TildeQ>;
}  // namespace

template class Filters::Hypercube<3, ff_tags>;
template class Filters::None<3, ff_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, ff_tags>;
