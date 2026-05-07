// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
using m1grey_tags =
    tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<
                   Frame::Inertial, neutrinos::ElectronNeutrinos<1>>,
               RadiationTransport::M1Grey::Tags::TildeS<
                   Frame::Inertial, neutrinos::ElectronNeutrinos<1>>>;
}  // namespace

template class Filters::Hypercube<3, m1grey_tags>;
template class Filters::None<3, m1grey_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, m1grey_tags>;
