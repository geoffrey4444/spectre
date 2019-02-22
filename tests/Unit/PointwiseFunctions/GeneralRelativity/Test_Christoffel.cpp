// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t Dim, IndexType Index, typename DataType>
void test_christoffel(const DataType& used_for_size) {
  tnsr::abb<DataType, Dim, Frame::Inertial, Index> (*f)(
      const tnsr::abb<DataType, Dim, Frame::Inertial, Index>&) =
      &gr::christoffel_first_kind<Dim, Frame::Inertial, Index, DataType>;
  pypp::check_with_random_values<1>(f, "TestFunctions",
                                    "christoffel_first_kind", {{{-10.0, 10.0}}},
                                    used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Christoffel",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");
  const DataVector dv(5);
  test_christoffel<1, IndexType::Spatial>(dv);
  test_christoffel<2, IndexType::Spatial>(dv);
  test_christoffel<3, IndexType::Spatial>(dv);
  test_christoffel<1, IndexType::Spacetime>(dv);
  test_christoffel<2, IndexType::Spacetime>(dv);
  test_christoffel<3, IndexType::Spacetime>(dv);
  test_christoffel<1, IndexType::Spatial>(0.);
  test_christoffel<2, IndexType::Spatial>(0.);
  test_christoffel<3, IndexType::Spatial>(0.);
  test_christoffel<1, IndexType::Spacetime>(0.);
  test_christoffel<2, IndexType::Spacetime>(0.);
  test_christoffel<3, IndexType::Spacetime>(0.);

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  CHECK(gr::Tags::SpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                       DataVector>::name() ==
        "SpacetimeChristoffelFirstKind");
  CHECK(gr::Tags::SpacetimeChristoffelSecondKindCompute<3, Frame::Inertial,
                                                        DataVector>::name() ==
        "SpacetimeChristoffelSecondKind");
  CHECK(
      gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                          DataVector>::name() ==
      "TraceSpacetimeChristoffelFirstKind");

  // Second, check that the compute items correctly compute their results
  DataVector test_vector{5.0, 4.0};
  auto deriv_spacetime_metric =
      make_with_value<tnsr::abb<DataVector, 3, Frame::Inertial>>(test_vector,
                                                                 0.0);
  get<0, 0, 0>(deriv_spacetime_metric) = 0.5;
  get<0, 0, 1>(deriv_spacetime_metric) = 0.1;
  get<0, 0, 2>(deriv_spacetime_metric) = 0.2;
  get<0, 0, 3>(deriv_spacetime_metric) = 0.3;
  get<0, 1, 1>(deriv_spacetime_metric) = 0.4;
  get<0, 1, 2>(deriv_spacetime_metric) = 0.2;
  get<0, 1, 3>(deriv_spacetime_metric) = 0.1;
  get<0, 2, 2>(deriv_spacetime_metric) = 0.3;
  get<0, 2, 3>(deriv_spacetime_metric) = 0.1;
  get<0, 3, 3>(deriv_spacetime_metric) = 0.2;

  get<1, 0, 0>(deriv_spacetime_metric) = -0.1;
  get<1, 0, 1>(deriv_spacetime_metric) = 0.1;
  get<1, 0, 2>(deriv_spacetime_metric) = -0.3;
  get<1, 0, 3>(deriv_spacetime_metric) = 0.3;
  get<1, 1, 1>(deriv_spacetime_metric) = -0.2;
  get<1, 1, 2>(deriv_spacetime_metric) = 0.2;
  get<1, 1, 3>(deriv_spacetime_metric) = -0.4;
  get<1, 2, 2>(deriv_spacetime_metric) = 0.4;
  get<1, 2, 3>(deriv_spacetime_metric) = -0.5;
  get<1, 3, 3>(deriv_spacetime_metric) = 0.5;

  get<2, 0, 0>(deriv_spacetime_metric) = 0.6;
  get<2, 0, 1>(deriv_spacetime_metric) = 0.5;
  get<2, 0, 2>(deriv_spacetime_metric) = 0.4;
  get<2, 0, 3>(deriv_spacetime_metric) = 0.3;
  get<2, 1, 1>(deriv_spacetime_metric) = 0.2;
  get<2, 1, 2>(deriv_spacetime_metric) = 0.1;
  get<2, 1, 3>(deriv_spacetime_metric) = 0.2;
  get<2, 2, 2>(deriv_spacetime_metric) = 0.3;
  get<2, 2, 3>(deriv_spacetime_metric) = 0.4;
  get<2, 3, 3>(deriv_spacetime_metric) = 0.5;

  get<3, 0, 0>(deriv_spacetime_metric) = -0.5;
  get<3, 0, 1>(deriv_spacetime_metric) = 0.1;
  get<3, 0, 2>(deriv_spacetime_metric) = -0.2;
  get<3, 0, 3>(deriv_spacetime_metric) = 0.3;
  get<3, 1, 1>(deriv_spacetime_metric) = -0.4;
  get<3, 1, 2>(deriv_spacetime_metric) = 0.2;
  get<3, 1, 3>(deriv_spacetime_metric) = -0.1;
  get<3, 2, 2>(deriv_spacetime_metric) = 0.3;
  get<3, 2, 3>(deriv_spacetime_metric) = -0.1;
  get<3, 3, 3>(deriv_spacetime_metric) = 0.2;

  auto inverse_spacetime_metric =
      make_with_value<tnsr::AA<DataVector, 3, Frame::Inertial>>(test_vector,
                                                                0.0);
  get<0, 0>(inverse_spacetime_metric) = -1.5;
  get<0, 1>(inverse_spacetime_metric) = 0.1;
  get<0, 2>(inverse_spacetime_metric) = 0.2;
  get<0, 3>(inverse_spacetime_metric) = 0.3;
  get<1, 1>(inverse_spacetime_metric) = 1.4;
  get<1, 2>(inverse_spacetime_metric) = 0.2;
  get<1, 3>(inverse_spacetime_metric) = 0.1;
  get<2, 2>(inverse_spacetime_metric) = 1.3;
  get<2, 3>(inverse_spacetime_metric) = 0.1;
  get<3, 3>(inverse_spacetime_metric) = 1.2;

  const auto& christoffel_first_kind =
      gr::christoffel_first_kind<3, Frame::Inertial, IndexType::Spacetime,
                                 DataVector>(deriv_spacetime_metric);
  const auto& christoffel_second_kind =
      raise_or_lower_first_index<DataVector,
                                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>(
          christoffel_first_kind, inverse_spacetime_metric);
  const auto& trace_christoffel_first_kind =
      trace_last_indices<DataVector,
                         SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                         SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>(
          christoffel_first_kind, inverse_spacetime_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::DerivativesOfSpacetimeMetric<3, Frame::Inertial,
                                                 DataVector>,
          gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>>,
      db::AddComputeTags<gr::Tags::SpacetimeChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::SpacetimeChristoffelSecondKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>>>(
      deriv_spacetime_metric, inverse_spacetime_metric);
  CHECK(db::get<gr::Tags::SpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                        DataVector>>(box) ==
        christoffel_first_kind);
  CHECK(db::get<gr::Tags::SpacetimeChristoffelSecondKind<3, Frame::Inertial,
                                                         DataVector>>(box) ==
        christoffel_second_kind);
  CHECK(db::get<gr::Tags::TraceSpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                             DataVector>>(
            box) == trace_christoffel_first_kind);
}
