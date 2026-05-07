// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
using valencia_tags =
    tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
               grmhd::ValenciaDivClean::Tags::TildeYe,
               grmhd::ValenciaDivClean::Tags::TildeTau,
               grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
               grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
               grmhd::ValenciaDivClean::Tags::TildePhi>;
}  // namespace

template class Filters::Hypercube<3, valencia_tags>;
template class Filters::None<3, valencia_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, valencia_tags>;
