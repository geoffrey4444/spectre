// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/Structure/OrientationMap.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

/*!
 * \brief No. Brief yourself.
 */
class Wedge final : public ShapeMapTransitionFunction {
  struct Surface {
    double radius{};
    double sphericity{};
    template <typename T>
    T distance(const std::array<T, 3>& coords) const;

    void pup(PUP::er& p);
  };

 public:
  explicit Wedge() = default;

  Wedge(double inner_radius, double outer_radius, double inner_sphericity,
        double outer_sphericity, OrientationMap<3> orientation_map);

  double operator()(const std::array<double, 3>& source_coords) const override;
  DataVector operator()(
      const std::array<DataVector, 3>& source_coords) const override;

  std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double distorted_radius) const override;

  double map_over_radius(
      const std::array<double, 3>& source_coords) const override;
  DataVector map_over_radius(
      const std::array<DataVector, 3>& source_coords) const override;

  std::array<double, 3> gradient(
      const std::array<double, 3>& source_coords) const override;
  std::array<DataVector, 3> gradient(
      const std::array<DataVector, 3>& source_coords) const override;

  WRAPPED_PUPable_decl_template(Wedge);
  explicit Wedge(CkMigrateMessage* const msg);
  void pup(PUP::er& p) override;

  std::unique_ptr<ShapeMapTransitionFunction> get_clone() const override {
    return std::make_unique<Wedge>(*this);
  }

  bool operator==(const ShapeMapTransitionFunction& other) const override;
  bool operator!=(const ShapeMapTransitionFunction& other) const override;

 private:
  template <typename T>
  T call_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  T map_over_radius_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  std::array<T, 3> gradient_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  void check_distances(const std::array<T, 3>& mag) const;

  Surface inner_surface_{};
  Surface outer_surface_{};
  OrientationMap<3> orientation_map_{};
  static constexpr double eps_ = std::numeric_limits<double>::epsilon() * 100;
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
