// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
using st_tags =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
               CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
               CurvedScalarWave::Tags::Phi<3, Frame::Inertial>>;
}  // namespace

template class Filters::Hypercube<3, st_tags>;
template class Filters::None<3, st_tags>;
template struct evolution::dg::Initialization::SpectralFilters<3, st_tags>;
