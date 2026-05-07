// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ghmhd_system =
    grmhd::GhValenciaDivClean::System<RadiationTransport::NoNeutrinos::System>;
using ghmhd_tags = typename ghmhd_system::variables_tag::tags_list;
}  // namespace

template class Filters::Hypercube<3, ghmhd_tags>;
template class Filters::None<3, ghmhd_tags>;
template class Filters::SphericalShell<ghmhd_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, ghmhd_tags>;
