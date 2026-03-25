// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <pup.h>
#include <string>

#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace {

double sqr(const double x) { return x * x; }

template <size_t N>
using LocalMatrix = std::array<std::array<double, N>, N>;

LocalMatrix<3> multiply_transpose(const LocalMatrix<3>& a,
                                  const LocalMatrix<3>& b) {
  LocalMatrix<3> result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          result[i][j] += a[k][i] * b[l][j] * (k <= l ? b[k][l] : b[l][k]);
        }
      }
    }
  }
  return result;
}

}  // namespace

namespace gr::Solutions {

TeukolskyWave::TeukolskyWave(CkMigrateMessage* /*msg*/) {}

TeukolskyWave::TeukolskyWave(double amplitude, const int mode,
                             std::string parity, std::string direction,
                             std::array<double, 3> center, const double radius,
                             const double width,
                             const Options::Context& context)
    : amplitude_(amplitude),
      mode_(mode),
      parity_(std::move(parity)),
      direction_(std::move(direction)),
      center_(center),
      radius_(radius),
      width_(width) {
  if (mode_ < -2 or mode_ > 2) {
    PARSE_ERROR(context, "Mode must lie between -2 and 2, inclusive.");
  }
  if (parity_ != "even" and parity_ != "odd") {
    PARSE_ERROR(context, "Parity must be either 'even' or 'odd'.");
  }
  if (direction_ != "outgoing" and direction_ != "ingoing") {
    PARSE_ERROR(context, "Direction must be either 'outgoing' or 'ingoing'.");
  }
  if (width_ <= 0.0) {
    PARSE_ERROR(context, "Width must be positive.");
  }
}

void TeukolskyWave::pup(PUP::er& p) {
  p | amplitude_;
  p | mode_;
  p | parity_;
  p | direction_;
  p | center_;
  p | radius_;
  p | width_;
}

TeukolskyWave::PointwiseData TeukolskyWave::pointwise_metric(
    const double x, const double y, const double z, const double t,
    const double amplitude, const int mode, const bool even_parity,
    const bool ingoing, const std::array<double, 3>& center,
    const double radius, const double width) {
  PointwiseData result{};
  for (size_t i = 0; i < 3; ++i) {
    result.spatial_metric[i][i] = 1.0;
  }

  const std::array<double, 3> c{{x - center[0], y - center[1], z - center[2]}};
  const double r = sqrt(sqr(c[0]) + sqr(c[1]) + sqr(c[2]));
  if (r <= 1.0e-14) {
    return result;
  }
  const double rho = sqrt(sqr(c[0]) + sqr(c[1]));
  const double costh = c[2] / r;
  const double sinth = rho / r;
  const double cosph = rho > 1.0e-14 ? c[0] / rho : 1.0;
  const double sinph = rho > 1.0e-14 ? c[1] / rho : 0.0;
  const double sin2phi = 2.0 * sinph * cosph;
  const double cos2phi = sqr(cosph) - sqr(sinph);

  double y_profile = r - radius;
  y_profile += ingoing ? t : -t;
  const double m2w = -2.0 / sqr(width);
  const double f0 = amplitude * exp(-sqr(y_profile) / sqr(width));
  const double f1 = m2w * y_profile * f0;
  const double f2 = m2w * (f0 + y_profile * f1);
  const double f3 = m2w * (2.0 * f1 + y_profile * f2);
  const double f4 = m2w * (3.0 * f2 + y_profile * f3);
  const double f5 = m2w * (4.0 * f3 + y_profile * f4);

  LocalMatrix<3> h_sph{};
  LocalMatrix<3> hdot_sph{};

  if (even_parity) {
    double frr = 0.0;
    double frth = 0.0;
    double frph = 0.0;
    double fthph = 0.0;
    double f1thth = 0.0;
    double f2thth = 0.0;
    double f1phph = 0.0;
    double f2phph = 0.0;
    switch (mode) {
      case -2:
        frr = sqr(sinth) * sin2phi;
        frth = sinth * costh * sin2phi;
        frph = sinth * cos2phi;
        f1thth = (1.0 + sqr(costh)) * sin2phi;
        f2thth = -sin2phi;
        fthph = -costh * cos2phi;
        f1phph = -f1thth;
        f2phph = sqr(costh) * sin2phi;
        break;
      case -1:
        frr = 2.0 * sinth * costh * sinph;
        frth = (sqr(costh) - sqr(sinth)) * sinph;
        frph = costh * cosph;
        f1thth = -2.0 * sinth * costh * sinph;
        f2thth = 0.0;
        fthph = sinth * cosph;
        f1phph = -f1thth;
        f2phph = -2.0 * sinth * costh * sinph;
        break;
      case 0:
        frr = 2.0 - 3.0 * sqr(sinth);
        frth = -3.0 * sinth * costh;
        frph = 0.0;
        f1thth = 3.0 * sqr(sinth);
        f2thth = -1.0;
        fthph = 0.0;
        f1phph = -f1thth;
        f2phph = 3.0 * sqr(sinth) - 1.0;
        break;
      case 1:
        frr = 2.0 * sinth * costh * cosph;
        frth = (sqr(costh) - sqr(sinth)) * cosph;
        frph = -costh * sinph;
        f1thth = -2.0 * sinth * costh * cosph;
        f2thth = 0.0;
        fthph = -sinth * sinph;
        f1phph = -f1thth;
        f2phph = -2.0 * sinth * costh * cosph;
        break;
      case 2:
        frr = sqr(sinth) * cos2phi;
        frth = sinth * costh * cos2phi;
        frph = -sinth * sin2phi;
        f1thth = (1.0 + sqr(costh)) * cos2phi;
        f2thth = -cos2phi;
        fthph = costh * sin2phi;
        f1phph = -f1thth;
        f2phph = sqr(costh) * cos2phi;
        break;
      default:
        ERROR("Unsupported Teukolsky mode");
    }

    const double a = 3.0 * (f2 + (-3.0 * f1 + 3.0 * f0 / r) / r) / (r * r * r);
    const double b =
        -(-f3 + (3.0 * f2 + (-6.0 * f1 + 6.0 * f0 / r) / r) / r) / (r * r);
    const double c_coef =
        0.25 *
        (f4 +
         (-2.0 * f3 + (9.0 * f2 + (-21.0 * f1 + 21.0 * f0 / r) / r) / r) / r) /
        r;
    h_sph[0][0] += a * frr;
    h_sph[1][0] += r * b * frth;
    h_sph[0][1] = h_sph[1][0];
    h_sph[2][0] += r * b * frph * sinth;
    h_sph[0][2] = h_sph[2][0];
    h_sph[1][1] += r * r * (c_coef * f1thth + a * f2thth);
    h_sph[2][1] += r * r * (a - 2.0 * c_coef) * fthph * sinth;
    h_sph[1][2] = h_sph[2][1];
    h_sph[2][2] += r * r * (c_coef * f1phph + a * f2phph) * sqr(sinth);

    const double sign = ingoing ? 1.0 : -1.0;
    const double dt_a =
        sign * 3.0 * (f3 + (-3.0 * f2 + 3.0 * f1 / r) / r) / (r * r * r);
    const double dt_b =
        -sign * (-f4 + (3.0 * f3 + (-6.0 * f2 + 6.0 * f1 / r) / r) / r) /
        (r * r);
    const double dt_c =
        sign * 0.25 *
        (f5 +
         (-2.0 * f4 + (9.0 * f3 + (-21.0 * f2 + 21.0 * f1 / r) / r) / r) / r) /
        r;
    hdot_sph[0][0] += dt_a * frr;
    hdot_sph[1][0] += r * dt_b * frth;
    hdot_sph[0][1] = hdot_sph[1][0];
    hdot_sph[2][0] += r * dt_b * frph * sinth;
    hdot_sph[0][2] = hdot_sph[2][0];
    hdot_sph[1][1] += r * r * (dt_c * f1thth + dt_a * f2thth);
    hdot_sph[2][1] += r * r * (dt_a - 2.0 * dt_c) * fthph * sinth;
    hdot_sph[1][2] = hdot_sph[2][1];
    hdot_sph[2][2] += r * r * (dt_c * f1phph + dt_a * f2phph) * sqr(sinth);
  } else {
    double drth = 0.0;
    double drph = 0.0;
    double dthth = 0.0;
    double dthph = 0.0;
    double dphph = 0.0;
    switch (mode) {
      case -2:
        drth = 4.0 * sinth * sin2phi;
        drph = 4.0 * sinth * costh * cos2phi;
        dthth = -2.0 * costh * sin2phi;
        dthph = -(2.0 - sqr(sinth)) * cos2phi;
        dphph = 2.0 * costh * sin2phi;
        break;
      case -1:
        drth = -2.0 * costh * sinph;
        drph = -2.0 * (sqr(costh) - sqr(sinth)) * cosph;
        dthth = -sinth * sinph;
        dthph = -costh * sinth * cosph;
        dphph = sinth * sinph;
        break;
      case 0:
        drth = 0.0;
        drph = -4.0 * costh * sinth;
        dthth = 0.0;
        dthph = -sqr(sinth);
        dphph = 0.0;
        break;
      case 1:
        drth = -2.0 * costh * cosph;
        drph = 2.0 * (sqr(costh) - sqr(sinth)) * sinph;
        dthth = -sinth * cosph;
        dthph = costh * sinth * sinph;
        dphph = sinth * cosph;
        break;
      case 2:
        drth = 4.0 * sinth * cos2phi;
        drph = -4.0 * sinth * costh * sin2phi;
        dthth = -2.0 * costh * cos2phi;
        dthph = (2.0 - sqr(sinth)) * sin2phi;
        dphph = 2.0 * costh * cos2phi;
        break;
      default:
        ERROR("Unsupported Teukolsky mode");
    }

    const double k = (f2 + (-3.0 * f1 + 3.0 * f0 / r) / r) / (r * r);
    const double l =
        (-f3 + (2.0 * f2 + (-3.0 * f1 + 3.0 * f0 / r) / r) / r) / r;
    h_sph[1][0] += r * k * drth;
    h_sph[0][1] = h_sph[1][0];
    h_sph[2][0] += r * k * drph * sinth;
    h_sph[0][2] = h_sph[2][0];
    h_sph[1][1] += r * r * l * dthth;
    h_sph[2][1] += r * r * l * dthph * sinth;
    h_sph[1][2] = h_sph[2][1];
    h_sph[2][2] += r * r * l * dphph * sqr(sinth);

    const double sign = ingoing ? 1.0 : -1.0;
    const double dt_k = sign * (f3 + (-3.0 * f2 + 3.0 * f1 / r) / r) / (r * r);
    const double dt_l =
        sign * (-f4 + (2.0 * f3 + (-3.0 * f2 + 3.0 * f1 / r) / r) / r) / r;
    hdot_sph[1][0] += r * dt_k * drth;
    hdot_sph[0][1] = hdot_sph[1][0];
    hdot_sph[2][0] += r * dt_k * drph * sinth;
    hdot_sph[0][2] = hdot_sph[2][0];
    hdot_sph[1][1] += r * r * dt_l * dthth;
    hdot_sph[2][1] += r * r * dt_l * dthph * sinth;
    hdot_sph[1][2] = hdot_sph[2][1];
    hdot_sph[2][2] += r * r * dt_l * dphph * sqr(sinth);
  }

  LocalMatrix<3> linv{};
  linv[0][0] = sinth * cosph;
  linv[0][1] = sinth * sinph;
  linv[0][2] = costh;
  linv[1][0] = costh * cosph / r;
  linv[1][1] = costh * sinph / r;
  linv[1][2] = -sinth / r;
  const double sinth_for_phi = std::max(sinth, 1.0e-14);
  linv[2][0] = -sinph / (r * sinth_for_phi);
  linv[2][1] = cosph / (r * sinth_for_phi);
  linv[2][2] = 0.0;

  const auto h_cart = multiply_transpose(linv, h_sph);
  const auto hdot_cart = multiply_transpose(linv, hdot_sph);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result.spatial_metric[i][j] += h_cart[i][j];
      result.dt_spatial_metric[i][j] = hdot_cart[i][j];
    }
  }
  return result;
}

bool operator==(const TeukolskyWave& lhs, const TeukolskyWave& rhs) {
  return lhs.amplitude() == rhs.amplitude() and lhs.mode() == rhs.mode() and
         lhs.parity() == rhs.parity() and lhs.direction() == rhs.direction() and
         lhs.center() == rhs.center() and lhs.radius() == rhs.radius() and
         lhs.width() == rhs.width();
}

bool operator!=(const TeukolskyWave& lhs, const TeukolskyWave& rhs) {
  return not(lhs == rhs);
}

}  // namespace gr::Solutions
