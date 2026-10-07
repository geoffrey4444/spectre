// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/RadialSourceProjection.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace {
std::pair<double, double> legendre_pair(const size_t degree, const double x) {
  double previous = 1.0;
  double current = x;
  if (degree == 0) {
    return {previous, 0.0};
  }
  for (size_t k = 2; k <= degree; ++k) {
    const double next =
        ((2.0 * k - 1.0) * x * current - (k - 1.0) * previous) / k;
    previous = current;
    current = next;
  }
  return {current, previous};
}

std::pair<DataVector, DataVector> high_lobatto(const size_t points) {
  DataVector nodes(points, 0.0);
  DataVector weights(points, 0.0);
  nodes[0] = -1.0;
  nodes[points - 1] = 1.0;
  const size_t degree = points - 1;
  for (size_t i = 1; i < points / 2; ++i) {
    double x = -std::cos(std::numbers::pi * i / degree);
    bool converged = false;
    for (size_t iteration = 0; iteration < 60; ++iteration) {
      const auto [p, previous] = legendre_pair(degree, x);
      const double derivative = degree * (x * p - previous) / (x * x - 1.0);
      const double second =
          (2.0 * x * derivative - degree * (degree + 1.0) * p) / (1.0 - x * x);
      const double update = derivative / second;
      x -= update;
      if (std::abs(update) < 4.0 * std::numeric_limits<double>::epsilon()) {
        converged = true;
        break;
      }
    }
    if (not converged) {
      ERROR("Temporary radial Lobatto nodes did not converge.");
    }
    nodes[i] = x;
    nodes[points - 1 - i] = -x;
  }
  for (size_t i = 0; i < points; ++i) {
    const double p = legendre_pair(degree, nodes[i]).first;
    weights[i] = 2.0 / (points * degree * p * p);
    if (not std::isfinite(weights[i]) or weights[i] <= 0.0 or
        (i > 0 and nodes[i] <= nodes[i - 1])) {
      ERROR("Invalid temporary radial Lobatto quadrature.");
    }
  }
  return {std::move(nodes), std::move(weights)};
}
}  // namespace

namespace gh {
RadialSourceProjection::RadialSourceProjection(const size_t radial_points,
                                               const size_t evaluation_points)
    : radial_points_(radial_points), evaluation_points_(evaluation_points) {
  if (radial_points < 2 or radial_points > 20 or
      (evaluation_points != radial_points and evaluation_points != 32 and
       evaluation_points != 40)) {
    ERROR(
        "Radial source projection requires evolved Nr2..20 and evaluation "
        "Nr32,40,or the original extent for a same-grid control.");
  }
  const auto& old_nodes =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          radial_points);
  if (evaluation_points == radial_points) {
    evaluation_nodes_ = old_nodes;
    evaluation_weights_ =
        Spectral::quadrature_weights<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto>(
            radial_points);
  } else {
    auto high = high_lobatto(evaluation_points);
    evaluation_nodes_ = std::move(high.first);
    evaluation_weights_ = std::move(high.second);
  }
  prolongation_ =
      Spectral::interpolation_matrix<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto>(
          radial_points, evaluation_nodes_);
  restriction_ = Matrix(radial_points, evaluation_points, 0.0);
  std::vector<DataVector> old_basis(radial_points, DataVector(radial_points));
  for (size_t k = 0; k < radial_points; ++k) {
    for (size_t i = 0; i < radial_points; ++i) {
      old_basis[k][i] = legendre_pair(k, old_nodes[i]).first;
    }
  }
  if (evaluation_points == radial_points) {
    for (size_t i = 0; i < radial_points; ++i) {
      restriction_(i, i) = 1.0;
    }
  } else {
    // High quadrature integrates every retained-basis / high-polynomial
    // product exactly because retained degree is below the high top degree.
    for (size_t k = 0; k < radial_points; ++k) {
      for (size_t j = 0; j < evaluation_points; ++j) {
        const double coefficient = (2.0 * k + 1.0) / 2.0 *
                                   evaluation_weights_[j] *
                                   legendre_pair(k, evaluation_nodes_[j]).first;
        for (size_t i = 0; i < radial_points; ++i) {
          restriction_(i, j) += old_basis[k][i] * coefficient;
        }
      }
    }
  }

  const auto& D =
      Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(
          radial_points);
  Matrix E(2, radial_points, 0.0);
  E(0, 0) = 1.0;
  for (size_t j = 0; j < radial_points; ++j) {
    E(1, j) = D(0, j) / ((radial_points - 1.0) * (radial_points - 1.0));
  }
  Matrix dual(radial_points, 2, 0.0);
  for (size_t i = 0; i < radial_points; ++i) {
    for (size_t j = 0; j < radial_points; ++j) {
      double inverse_mass = 0.0;
      for (size_t k = 0; k < radial_points; ++k) {
        inverse_mass +=
            old_basis[k][i] * (2.0 * k + 1.0) / 2.0 * old_basis[k][j];
      }
      for (size_t a = 0; a < 2; ++a) {
        dual(i, a) += inverse_mass * E(a, j);
      }
    }
  }
  Matrix gram(2, 2, 0.0);
  for (size_t a = 0; a < 2; ++a) {
    for (size_t b = 0; b < 2; ++b) {
      for (size_t j = 0; j < radial_points; ++j) {
        gram(a, b) += E(a, j) * dual(j, b);
      }
    }
  }
  const double determinant = gram(0, 0) * gram(1, 1) - gram(0, 1) * gram(1, 0);
  if (not std::isfinite(determinant) or determinant <= 0.0) {
    ERROR("Singular endpoint constraints in radial source projection.");
  }
  const std::array<std::array<double, 2>, 2> inverse{
      {{{gram(1, 1) / determinant, -gram(0, 1) / determinant}},
       {{-gram(1, 0) / determinant, gram(0, 0) / determinant}}}};
  preserve_inner_ = Matrix(radial_points, radial_points, 0.0);
  for (size_t i = 0; i < radial_points; ++i) {
    for (size_t j = 0; j < radial_points; ++j) {
      double removed = 0.0;
      for (size_t a = 0; a < 2; ++a) {
        for (size_t b = 0; b < 2; ++b) {
          removed += dual(i, a) * inverse[a][b] * E(b, j);
        }
      }
      preserve_inner_(i, j) = (i == j ? 1.0 : 0.0) - removed;
    }
  }
}

DataVector RadialSourceProjection::apply(const Matrix& matrix,
                                         const DataVector& values,
                                         const size_t components) {
  if (components == 0 or values.size() % (components * matrix.columns()) != 0) {
    ERROR("Invalid component-major radial-source array size.");
  }
  const size_t angular = values.size() / (components * matrix.columns());
  DataVector result(components * angular * matrix.rows(), 0.0);
  for (size_t c = 0; c < components; ++c) {
    for (size_t a = 0; a < angular; ++a) {
      for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.columns(); ++j) {
          result[(c * angular + a) * matrix.rows() + i] +=
              matrix(i, j) * values[(c * angular + a) * matrix.columns() + j];
        }
      }
    }
  }
  return result;
}

DataVector RadialSourceProjection::prolong(const DataVector& values,
                                           const size_t components) const {
  return apply(prolongation_, values, components);
}

DataVector RadialSourceProjection::source_increment(
    const DataVector& coarse_source, const DataVector& high_source,
    const size_t components, const bool preserve_inner) const {
  DataVector increment = apply(restriction_, high_source, components);
  if (increment.size() != coarse_source.size()) {
    ERROR("Coarse and high radial-source array sizes disagree.");
  }
  increment -= coarse_source;
  return preserve_inner ? apply(preserve_inner_, increment, components)
                        : increment;
}
}  // namespace gh
