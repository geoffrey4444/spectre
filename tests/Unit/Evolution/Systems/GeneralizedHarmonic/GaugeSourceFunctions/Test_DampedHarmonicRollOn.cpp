// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonicRollOn.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.tpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
using Kerr = gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>;
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<gh::gauges::GaugeCondition,
                   tmpl::list<gh::gauges::DampedHarmonicRollOn>>,
        tmpl::pair<evolution::initial_data::InitialData, tmpl::list<Kerr>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonicRollOn",
    "[Unit][Evolution]") {
  register_classes_with_charm<gh::gauges::DampedHarmonicRollOn, Kerr>();
  const auto gauge = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<gh::gauges::GaugeCondition>,
                                 Metavariables>(
          "DampedHarmonicRollOn:\n"
          "  InitialGauge:\n"
          "    AnalyticPrescription:\n"
          "      GeneralizedHarmonic(KerrSchild):\n"
          "        Mass: 1.0\n"
          "        Spin: [0.0, 0.0, 0.999]\n"
          "        Center: [0.0, 0.0, 0.0]\n"
          "        Velocity: [0.0, 0.0, 0.0]\n"
          "  RollOnStartTime: 2.0\n"
          "  RollOnTimeWindow: 10.0\n"
          "  SpatialDecayWidth: 17.0\n"
          "  Amplitudes: [1.0, 0.0, 1.0]\n"
          "  Exponents: [2, 2, 2]\n")
          ->get_clone());
  CHECK_FALSE(gauge->is_harmonic());

  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t size = mesh.number_of_grid_points();
  const auto logical = logical_coordinates(mesh);
  tnsr::I<DataVector, 3> coords{size};
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{size, 0.0};
  tnsr::II<DataVector, 3> inverse_spatial_metric{size, 0.0};
  tnsr::aa<DataVector, 3> metric{size, 0.0};
  get<0, 0>(metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    coords.get(i) = logical.get(i) + 3.0;
    inverse_jacobian.get(i, i) = 1.0;
    inverse_spatial_metric.get(i, i) = 1.0;
    metric.get(i + 1, i + 1) = 1.0;
  }
  const gh::gauges::AnalyticChristoffel initial{std::make_unique<Kerr>(
      1.0, std::array{0.0, 0.0, 0.999}, std::array{0.0, 0.0, 0.0},
      std::array{0.0, 0.0, 0.0})};
  tnsr::a<DataVector, 3> initial_h{size}, h{size};
  tnsr::ab<DataVector, 3> initial_dh{size}, dh{size};
  initial.gauge_and_spacetime_derivative(
      make_not_null(&initial_h), make_not_null(&initial_dh), mesh, 0.0, coords,
      inverse_jacobian, tmpl::list<Kerr>{});

  // Flat evolved fields have H_DH = dH_DH = 0. The stationary analytic gauge
  // still has nonzero H_init, so the temporal product-rule term is tested.
  for (const double time : {-1.0, 2.0, 7.0, 12.0, 22.0, 42.0}) {
    CAPTURE(time);
    gh::gauges::dispatch<tmpl::list<Kerr>>(
        make_not_null(&h), make_not_null(&dh), Scalar<DataVector>{size, 1.0},
        tnsr::I<DataVector, 3>{size, 0.0}, Scalar<DataVector>{size, 1.0},
        inverse_spatial_metric, tnsr::abb<DataVector, 3>{size, 0.0},
        Scalar<DataVector>{size, 0.0}, tnsr::i<DataVector, 3>{size, 0.0},
        metric, tnsr::iaa<DataVector, 3>{size, 0.0}, mesh, time, coords,
        inverse_jacobian, *gauge);
    const double roll =
        gh::gauges::DampedHarmonicGauge_detail::roll_on_function(time, 2.0,
                                                                 10.0);
    const double dt_roll =
        gh::gauges::DampedHarmonicGauge_detail::time_deriv_of_roll_on_function(
            time, 2.0, 10.0);
    for (size_t a = 0; a < 4; ++a) {
      CHECK_ITERABLE_APPROX(h.get(a), (1.0 - roll) * initial_h.get(a));
      CHECK_ITERABLE_APPROX(dh.get(0, a), -dt_roll * initial_h.get(a));
      for (size_t i = 0; i < 3; ++i) {
        CHECK_ITERABLE_APPROX(dh.get(i + 1, a),
                              (1.0 - roll) * initial_dh.get(i + 1, a));
      }
    }
  }
}
