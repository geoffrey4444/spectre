// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/SpECMultiPenalty.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"

namespace {
using Penalty = gh::BoundaryCorrections::SpECMultiPenalty;

Mesh<2> face_mesh(const size_t l) {
  return Mesh<2>{
      {l + 1, 2 * l + 1},
      {Spectral::Basis::SphericalHarmonic, Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
}

// Real sectoral harmonic of degree m and a constant.
DataVector data_on_sphere(const size_t l, const int m) {
  const auto angles = ylm::get_spherepack_cache(l).theta_phi_points();
  return 0.2 + pow(sin(angles[0]), m) * cos(m * angles[1]);
}

void check_transfer(const size_t receiver_l, const size_t donor_l, const int m,
                    const size_t kill, const bool killed,
                    const double normal_sign) {
  CAPTURE(receiver_l);
  CAPTURE(donor_l);
  CAPTURE(kill);
  const auto mesh = face_mesh(receiver_l);
  const size_t points = mesh.number_of_grid_points();
  DataVector local(
      points * Variables<Penalty::NativeTags>::number_of_independent_components,
      0.);
  Variables<Penalty::NativeTags> native(local.data(), local.size());
  auto& metric = get<Penalty::Metric>(native);
  get<0, 0>(metric) = -1.;
  for (size_t i = 0; i < 3; ++i) {
    metric.get(i + 1, i + 1) = 1.;
    get<Penalty::Jac>(native).get(i, i) = 1.;
    get<Penalty::InvJac>(native).get(i, i) = 1.;
  }
  get(get<gr::Tags::Lapse<DataVector>>(native)) = 1.;
  get(get<Penalty::Magnitude>(native)) = 2.;
  get<0>(get<Penalty::Normal>(native)) = normal_sign;

  const auto source = face_mesh(donor_l);
  DataVector donor(
      source.number_of_grid_points() *
          Variables<Penalty::Fields>::number_of_independent_components,
      0.);
  Variables<Penalty::Fields> fields(donor.data(), donor.size());
  get<0, 0>(get<Penalty::Pi>(fields)) = data_on_sphere(donor_l, m);

  const auto penalty = serialize_and_deserialize(Penalty{kill});
  const auto result = penalty.native_face_terms(local, donor, mesh, source, 5);
  // Only v-minus is incoming. Its penalty splits equally into Pi_tt and
  // -n_x Phi_xtt. The coefficient is (5-1)^2 * 2 / 2 = 16.
  Variables<Penalty::Fields> expected(points, 0.);
  auto& pi_tt = get<0, 0>(get<Penalty::Pi>(expected));
  pi_tt =
      8. * (killed ? DataVector(points, 0.2) : data_on_sphere(receiver_l, m));
  get<0, 0, 0>(get<Penalty::Phi>(expected)) = -normal_sign * pi_tt;
  CHECK_VARIABLES_APPROX(result, expected);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BoundaryCorrections.SpECMultiPenalty",
    "[Unit][Evolution]") {
  // Fine to coarse retains the point values of a mode above the receiver's L.
  check_transfer(4, 6, 6, 0, false, -1.);
  // Coarse to fine filters the interpolated data at the receiver's cutoff.
  check_transfer(5, 4, 4, 0, false, 1.);
  check_transfer(5, 4, 4, 2, true, 1.);
  // SpEC skips the incoming filter on a conforming interface.
  check_transfer(4, 4, 4, 2, false, -1.);
}
