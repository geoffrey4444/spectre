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
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
template <size_t Dim>
using tags_for_filter =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
               gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template class Filters::Hypercube<DIM(data), tags_for_filter<DIM(data)>>; \
  template class Filters::None<DIM(data), tags_for_filter<DIM(data)>>;      \
  template struct evolution::dg::Initialization::SpectralFilters<           \
      DIM(data), tags_for_filter<DIM(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

template class Filters::SphericalShell<tags_for_filter<3>>;

#undef DIM
#undef INSTANTIATE
