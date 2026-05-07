// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
using scalar_advection_tags = tmpl::list<ScalarAdvection::Tags::U>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                    \
  template class Filters::Hypercube<DIM(data), scalar_advection_tags>;         \
  template class Filters::None<DIM(data), scalar_advection_tags>;              \
  template struct evolution::dg::Initialization::SpectralFilters<              \
      DIM(data), scalar_advection_tags>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
