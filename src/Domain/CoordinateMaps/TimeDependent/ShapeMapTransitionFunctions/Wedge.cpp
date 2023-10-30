// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <pup.h>

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

#include "Parallel/Printf.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
template <typename T>
T Wedge::Surface::distance(const std::array<T, 3>& coords) const {
  // Short circuit if it's a sphere. Then the distance is trivially the radius
  // of this surface
  if (sphericity == 1.0) {
    return make_with_value<T>(coords[0], radius);
  }

  // D = R * [ (1 - s) / (sqrt(3) * cos(theta)) + s]
  // cos(theta) = z / r
  return radius *
         ((1.0 - sphericity) / (coords[2] * sqrt(3.0)) * magnitude(coords) +
          sphericity);
}

void Wedge::Surface::pup(PUP::er& p) {
  p | radius;
  p | sphericity;
}

Wedge::Wedge(const double inner_radius, const double outer_radius,
             const double inner_sphericity, const double outer_sphericity,
             OrientationMap<3> orientation_map)
    : inner_surface_(Surface{inner_radius, inner_sphericity}),
      outer_surface_(Surface{outer_radius, outer_sphericity}),
      orientation_map_(std::move(orientation_map)) {}

double Wedge::operator()(const std::array<double, 3>& source_coords) const {
  return call_impl<double>(source_coords);
}
DataVector Wedge::operator()(
    const std::array<DataVector, 3>& source_coords) const {
  return call_impl<DataVector>(source_coords);
}

std::optional<double> Wedge::original_radius_over_radius(
    const std::array<double, 3>& target_coords, double distorted_radius) const {
  const double radius = magnitude(target_coords);
  if (equal_within_roundoff(radius, 0.0) or
      equal_within_roundoff(distorted_radius, 0.0)) {
    return std::nullopt;
  }

  const std::array<double, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), target_coords);
  const double outer_distance = outer_surface_.distance(rotated_coords);
  const double distance_difference =
      outer_distance - inner_surface_.distance(rotated_coords);

  // TODO: Comment functional form
  const double a = radius;
  const double c = -distance_difference / distorted_radius;
  const double b = -c - outer_distance;

  // QUESTION: Is this the right one to use?
  std::optional<std::array<double, 2>> roots = real_roots(a, b, c);

  if (roots.has_value()) {
    // Parallel::printf(
    //     "a = %.16f\n"
    //     "b = %.16f\n"
    //     "c = %.16f\n"
    //     "sqrt(b^2 - 4ac) = %.16f\n"
    //     "Roots: %s\n\n",
    //     a, b, c, sqrt(square(b) - 4 * a * c), roots.value());
    return roots.value()[1];
  } else {
    return std::nullopt;
  }
}

double Wedge::map_over_radius(
    const std::array<double, 3>& source_coords) const {
  return map_over_radius_impl<double>(source_coords);
}
DataVector Wedge::map_over_radius(
    const std::array<DataVector, 3>& source_coords) const {
  return map_over_radius_impl<DataVector>(source_coords);
}

std::array<double, 3> Wedge::gradient(
    const std::array<double, 3>& source_coords) const {
  return gradient_impl<double>(source_coords);
}
std::array<DataVector, 3> Wedge::gradient(
    const std::array<DataVector, 3>& source_coords) const {
  return gradient_impl<DataVector>(source_coords);
}

template <typename T>
T Wedge::call_impl(const std::array<T, 3>& source_coords) const {
  const std::array<T, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), source_coords);
  check_distances(rotated_coords);
  T outer_distance = outer_surface_.distance(rotated_coords);

  return (outer_distance - magnitude(rotated_coords)) /
         (outer_distance - inner_surface_.distance(rotated_coords));
}

template <typename T>
T Wedge::map_over_radius_impl(const std::array<T, 3>& source_coords) const {
  const std::array<T, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), source_coords);
  check_distances(rotated_coords);
  const T radius = magnitude(rotated_coords);
  const T outer_distance = outer_surface_.distance(rotated_coords);

  return (outer_distance - radius) /
         ((outer_distance - inner_surface_.distance(rotated_coords)) * radius);
}

template <typename T>
std::array<T, 3> Wedge::gradient_impl(
    const std::array<T, 3>& source_coords) const {
  // If both surfaces are spherical then we short circuit because the distances
  // are constant and we only need to take a derivative of r.
  // (grad f)_i = -(x_i/r)/(D_out - D_in)
  const std::array<T, 3> rotated_coords =
      discrete_rotation(orientation_map_.inverse_map(), source_coords);
  check_distances(rotated_coords);
  if (inner_surface_.sphericity == 1.0 and outer_surface_.sphericity == 1.0) {
    const T one_over_denom = 1.0 / (magnitude(rotated_coords) *
                                    (outer_surface_.distance(rotated_coords) -
                                     inner_surface_.distance(rotated_coords)));

    return -source_coords * one_over_denom;
  }

  const T radius = magnitude(rotated_coords);

  const T one_over_radius = 1.0 / radius;
  T outer_distance = outer_surface_.distance(rotated_coords);
  const T one_over_denom =
      1.0 / (outer_distance - inner_surface_.distance(rotated_coords));
  T& outer_distance_minus_radius = outer_distance;
  outer_distance_minus_radius = outer_distance - radius;

  // Avoid roundoff if we are at outer boundary
  for (size_t i = 0; i < get_size(radius); i++) {
    if (equal_within_roundoff(get_element(outer_distance_minus_radius, i),
                              0.0)) {
      get_element(outer_distance_minus_radius, i) = 0.0;
    }
  }

  // Regardless of the sphericities below, we always need this factor in the
  // first term so we calculate it now.
  std::array<T, 3> result = -1.0 * rotated_coords * one_over_radius;

  const auto make_factor = [&one_over_radius](const Surface& surface) -> T {
    return (1.0 - surface.sphericity) * surface.radius / sqrt(3.0) *
           cube(one_over_radius);
  };

  // We can make some simplifications if either of the surfaces are spherical
  // because then the derivative of the distance is zero since it's constant. In
  // the first two branches, it's safe to assume the other sphericity isn't 1
  // because of the above check.
  // TODO: Add more comments below
  if (outer_surface_.sphericity == 1.0) {
    const T inner_surface_factor = make_factor(inner_surface_);

    for (size_t i = 0; i < 2; i++) {
      gsl::at(result, i) -= outer_distance_minus_radius * inner_surface_factor *
                            rotated_coords[2] * gsl::at(rotated_coords, i) *
                            one_over_denom;
    }

    result[2] += outer_distance_minus_radius * inner_surface_factor *
                 (square(radius) - square(rotated_coords[2])) * one_over_denom;
  } else if (inner_surface_.sphericity == 1.0) {
    std::array<T, 3> outer_deriv_distance =
        make_array<3>(make_factor(outer_surface_));
    for (size_t i = 0; i < 2; i++) {
      gsl::at(outer_deriv_distance, i) *=
          -1.0 * rotated_coords[2] * gsl::at(rotated_coords, i);
    }
    outer_deriv_distance[2] *= (square(radius) - square(rotated_coords[2]));

    result += outer_deriv_distance *
              (1.0 - outer_distance_minus_radius * one_over_denom);
  } else {
    const T inner_surface_factor = make_factor(inner_surface_);
    std::array<T, 3> outer_deriv_distance =
        make_array<3>(make_factor(outer_surface_));
    for (size_t i = 0; i < 2; i++) {
      gsl::at(outer_deriv_distance, i) *=
          -1.0 * rotated_coords[2] * gsl::at(rotated_coords, i);
    }
    outer_deriv_distance[2] *= (square(radius) - square(rotated_coords[2]));

    result += outer_deriv_distance;
    for (size_t i = 0; i < 3; i++) {
      if (i != 2) {
        gsl::at(result, i) -= outer_distance_minus_radius *
                              (gsl::at(outer_deriv_distance, i) -
                               (inner_surface_factor * rotated_coords[2] *
                                gsl::at(rotated_coords, i))) *
                              one_over_denom;
      } else {
        gsl::at(result, i) -= outer_distance_minus_radius *
                              (gsl::at(outer_deriv_distance, i) -
                               (inner_surface_factor *
                                (square(radius) - square(rotated_coords[2])))) *
                              one_over_denom;
      }
    }
  }

  // Finally, need one more factor of D_out - D_in in the denominator and to
  // rotate it back to the proper orientation
  return discrete_rotation(orientation_map_, result * one_over_denom);
}

bool Wedge::operator==(const ShapeMapTransitionFunction& other) const {
  if (dynamic_cast<const Wedge*>(&other) == nullptr) {
    return false;
  }
  return true;
}

bool Wedge::operator!=(const ShapeMapTransitionFunction& other) const {
  return not(*this == other);
}

// checks that the magnitudes are all between `r_min_` and `r_max_`
template <typename T>
void Wedge::check_distances([
    [maybe_unused]] const std::array<T, 3>& coords) const {
#ifdef SPECTRE_DEBUG
  const T mag = magnitude(coords);
  const T inner_distance = inner_surface_.distance(coords);
  const T outer_distance = outer_surface_.distance(coords);
  for (size_t i = 0; i < get_size(mag); ++i) {
    if (get_element(mag, i) + eps_ < get_element(inner_distance, i) or
        get_element(mag, i) - eps_ > get_element(outer_distance, i)) {
      ERROR(
          "The Wedge transition map was called with coordinates outside "
          "the set inner and outer surfaces. The inner radius and sphericity "
          "are (r="
          << inner_surface_.radius << ",s=" << inner_surface_.sphericity
          << ") and the outer radius and sphericity are (r="
          << outer_surface_.radius << ",s=" << outer_surface_.sphericity
          << "). The inner distance is " << get_element(inner_distance, i)
          << ", the outer distance is " << get_element(outer_distance, i)
          << ". The requested point has radius: " << get_element(mag, i));
    }
  }
#endif  // SPECTRE_DEBUG
}

void Wedge::pup(PUP::er& p) {
  ShapeMapTransitionFunction::pup(p);
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | inner_surface_;
    p | outer_surface_;
    p | orientation_map_;
  }
}

Wedge::Wedge(CkMigrateMessage* const msg) : ShapeMapTransitionFunction(msg) {}

PUP::able::PUP_ID Wedge::my_PUP_ID = 0;

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
