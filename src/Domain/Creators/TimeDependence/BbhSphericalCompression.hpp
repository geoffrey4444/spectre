// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
namespace CoordinateMaps {
namespace TimeDependent {
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace TimeDependent
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
namespace time_dependence {
/*!
 * \brief Applies a time-dependent spherical compression CoordinateMap to
 * the spherical blocks near each excision surface in a BinaryCompactObject
 * domain configured for evolving binary black holes.
 *
 * \details The idea of this map
 * is to move the excision surfaces outward, to compensate for the excision
 * surfaces' tendancy to fall inward as two black holes approach each other in
 * an inspiral or head-on merger. It applies a
 * domain::CoordinateMaps::TimeDependent::SphericalCompression to the blocks in
 * layer 1 of domain::Creators::BinaryCompactObject with both `ExciseInteriorA
 * == true` and `ExciseInteriorB == true`. A distinct CoordinateMap deforms the
 * blocks near black hole A and the blocks near black hole B. These blocks form
 * a layer of "spherical shells" around each excision surface. Note that the
 * options `InnerRadiusObjectA`, `InnerRadiusObjectB`, `OuterRadiusObjectA`,
 * `OuterRadiusObjectB`, `XCoordObjectA`, and `XCoordObjectB` must agree with
 * those options as given in for domain::Creators::BinaryCompactObject; this
 * ensures that the discontinuities in the Jacobians of the CoordinateMap occur
 * at the block boundaries. When composing with other time-dependent
 * CoordinateMaps (such as expansion and rotation), this should be the first
 * (innermost) map applied. The options `FunctionOfTimeA` and `FunctionOfTimeB`
 * name two scalar FunctionOfTime objects \f$\lambda_00^A\f$ and
 * \f$\lambda_{00}^B\f$ that determine the amount of compression in each block.
 *
 * \note This could be generalized in the future to include an option to only
 * deform one of the objects, as would be suitable for a black-hole/neutron-star
 * simulation, and it could also be generalized to include a more generic
 * deformation, such as not assuming the block boundaries are spheres and such
 * as including shape deformations as well as a spherical compression.
 */
template <size_t MeshDim>
class BbhSphericalCompression final : public TimeDependence<MeshDim> {
  static_assert(MeshDim == 3,
                "BbhSphericalCompression<MeshDim> undefined for MeshDim != 3");

 private:
  template <bool InteriorMap>
  using CoordinateMapA =
      domain::CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap>;
  using CoordinateMapB =
      domain::CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap>;

 public:
  using maps_list = tmpl::list<CoordinateMapA<true>, CoordinateMapA<false>,
                               CoordinateMapB<true>, CoordinateMapB<false>>;

  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };
  /// \brief The time interval for updates of the functions of time.
  struct InitialExpirationDeltaT {
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "The initial time interval for updates of the functions of time. If "
        "Auto, then the functions of time do not expire, nor can they be "
        "updated."};
  };
  /// \brief The time derivative of the compression for black hole A.
  struct CompressionVelocityA {
    using type = double;
    static constexpr Options::String help = {
        "The compression velocity of the map for black hole A."};
  };
  /// \brief The time derivative of the compression for black hole B.
  struct CompressionVelocityB {
    using type = double;
    static constexpr Options::String help = {
        "The compression velocity of the map for black hole B."};
  };
  /// \brief The name of the function of time to be added to the DataBox for
  /// black hole A.
  struct FunctionOfTimeA {
    using type = std::string;
    static constexpr Options::String help = {
        "Name of the compression factor function of time for black hole A."};
  };
  /// \brief The name of the function of time to be added to the DataBox for
  /// black hole B.
  struct FunctionOfTimeB {
    using type = std::string;
    static constexpr Options::String help = {
        "Name of the compression factor function of time for black hole B."};
  };

  // Options that must match the options given to the domain creator. Can this
  // be done in a smarter way than just giving the options all over again?
  struct InnerRadiusObjectA {
    using type = double;
    static constexpr Options::String help = {
        "Inner coordinate radius of Layer 1 for Object A."};
  };

  struct OuterRadiusObjectA {
    using type = double;
    static constexpr Options::String help = {
        "Outer coordinate radius of Layer 1 for Object A."};
  };

  struct XCoordObjectA {
    using type = double;
    static constexpr Options::String help = {
        "x-coordinate of center for Object A."};
  };

  struct InnerRadiusObjectB {
    using type = double;
    static constexpr Options::String help = {
        "Inner coordinate radius of Layer 1 for Object B."};
  };

  struct OuterRadiusObjectB {
    using type = double;
    static constexpr Options::String help = {
        "Outer coordinate radius of Layer 1 for Object B."};
  };

  struct XCoordObjectB {
    using type = double;
    static constexpr Options::String help = {
        "x-coordinate of the center for Object B."};
  };

  using MapForComposition = detail::generate_coordinate_map_t<
      domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
          CoordinateMapA<true>, CoordinateMapA<false>, CoordinateMapB<true>,
          CoordinateMapB<false>>>;

  using options =
      tmpl::list<InitialTime, InitialExpirationDeltaT, CompressionVelocityA,
                 CompressionVelocityB, InnerRadiusObjectA, OuterRadiusObjectA,
                 XCoordObjectA, InnerRadiusObjectB, OuterRadiusObjectB,
                 XCoordObjectB, FunctionOfTimeA, FunctionOfTimeB>;

  static constexpr Options::String help = {
      "A time-dependent compression of the layer-1 blocks surrounding each "
      "excision surface in a BinaryCompactObject domain for a binary black "
      "hole."};

  BbhSphericalCompression() = default;
  ~BbhSphericalCompression() override = default;
  BbhSphericalCompression(const BbhSphericalCompression&) = delete;
  BbhSphericalCompression(BbhSphericalCompression&&) noexcept = default;
  BbhSphericalCompression& operator=(const BbhSphericalCompression&) = delete;
  BbhSphericalCompression& operator=(BbhSphericalCompression&&) noexcept =
      default;

  BbhSphericalCompression(
      double initial_time, std::optional<double> initial_expiration_delta_t,
      double compression_velocity_a, compression_velocity_b,
      double inner_radius_object_a, double outer_radius_object_a,
      double x_coord_object_a, double inner_radius_object_b,
      double outer_radius_object_b, double x_coord_object_b,
      std::string function_of_time_a = "LambdaFactorA0",
      std::string function_of_time_a = "LambdaFactorB0") noexcept;

  auto get_clone() const noexcept
      -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps(size_t number_of_blocks) const noexcept
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  /// Returns the map for each block to be used in a composition of
  /// `TimeDependence`s.
  MapForComposition map_for_composition() const noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const BbhSphericalCompression<LocalDim>& lhs,
                         const BbhSphericalCompression<LocalDim>& rhs) noexcept;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  std::optional<double> initial_expiration_delta_t_{};
  double compression_velocity_a_{std::numeric_limits<double>::signaling_NaN()};
  double compression_velocity_b_{std::numeric_limits<double>::signaling_NaN()};

  // Store options that must agree with the values given to the DomainCreator
  double inner_radius_object_a_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_object_a_{std::numeric_limits<double>::signaling_NaN()};
  double x_coord_object_a_{std::numeric_limits<double>::signaling_NaN()};
  double inner_radius_object_b_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_object_b_{std::numeric_limits<double>::signaling_NaN()};
  double x_coord_object_b_{std::numeric_limits<double>::signaling_NaN()};

  std::string function_of_time_a_{};
  std::string function_of_time_b_{};
};

template <size_t Dim>
bool operator==(const BbhSphericalCompression<Dim>& lhs,
                const BbhSphericalCompression<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const BbhSphericalCompression<Dim>& lhs,
                const BbhSphericalCompression<Dim>& rhs) noexcept;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
