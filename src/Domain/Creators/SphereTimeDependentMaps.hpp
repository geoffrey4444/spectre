// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::sphere {
/*!
 * \brief Mass and spin necessary for calculating the \f$ Y_{lm} \f$
 * coefficients of a Kerr horizon of certain Boyer-Lindquist radius for the
 * shape map of the Sphere domain creator.
 */
struct KerrSchildFromBoyerLindquist {
  /// \brief The mass of the Kerr black hole.
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"The mass of the Kerr BH."};
  };
  /// \brief The dimensionless spin of the Kerr black hole.
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The dim'less spin of the Kerr BH."};
  };

  using options = tmpl::list<Mass, Spin>;

  static constexpr Options::String help = {
      "Conform to an ellipsoid of constant Boyer-Lindquist radius in "
      "Kerr-Schild coordinates. This Boyer-Lindquist radius is chosen as the "
      "value of the 'InnerRadius'. To conform to the outer Kerr horizon, "
      "choose an 'InnerRadius' of r_+ = M + sqrt(M^2-a^2)."};

  double mass{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> spin{std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN()};
};

// Label for shape map options
struct Spherical {};

// HACK: drop in from BinaryCompactObjectHelpers.hpp
// should probably extract this into its own separate unit of code
namespace detail {
// Convenience type alias
template <typename... Maps>
using gi_map = domain::CoordinateMap<Frame::Grid, Frame::Inertial, Maps...>;
template <typename... Maps>
using gd_map = domain::CoordinateMap<Frame::Grid, Frame::Distorted, Maps...>;
template <typename... Maps>
using di_map =
    domain::CoordinateMap<Frame::Distorted, Frame::Inertial, Maps...>;

// Code to compute all possible map combinations for map_list
template <typename List>
struct power_set {
  using rest = typename power_set<tmpl::pop_front<List>>::type;
  using type = tmpl::append<
      rest, tmpl::transform<rest, tmpl::lazy::push_front<
                                      tmpl::_1, tmpl::pin<tmpl::front<List>>>>>;
};
template <>
struct power_set<tmpl::list<>> {
  using type = tmpl::list<tmpl::list<>>;
};
template <typename SourceFrame, typename TargetFrame, typename Maps>
using produce_all_maps_helper =
    tmpl::wrap<tmpl::push_front<Maps, SourceFrame, TargetFrame>,
               domain::CoordinateMap>;
template <typename SourceFrame, typename TargetFrame, typename... Maps>
using produce_all_maps = tmpl::transform<
    tmpl::remove<typename power_set<tmpl::list<Maps...>>::type, tmpl::list<>>,
    tmpl::bind<produce_all_maps_helper, tmpl::pin<SourceFrame>,
               tmpl::pin<TargetFrame>, tmpl::_1>>;
}  // namespace detail

/*!
 * \brief This holds all options related to the time dependent maps of the
 * domain::creators::Sphere domain creator.
 *
 * \details Currently this class will only add a Shape map (and size
 * FunctionOfTime) to the domain. Other maps can be added as needed.
 *
 * \note This struct contains no information about what blocks the time
 * dependent maps will go in.
 */
struct TimeDependentMapOptions {
 private:
  template <typename SourceFrame, typename TargetFrame>
  using MapType =
      std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>;
  using IdentityMap = domain::CoordinateMaps::Identity<3>;
  // Time-dependent maps
  using ShapeMap = domain::CoordinateMaps::TimeDependent::Shape;
  using ExpansionMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
  using RotationMap = domain::CoordinateMaps::TimeDependent::Rotation<3>;

  template <typename SourceFrame, typename TargetFrame>
  using IdentityForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, IdentityMap>;

 public:
  using maps_list = tmpl::list<
      IdentityForComposition<Frame::Grid, Frame::Inertial>,
      IdentityForComposition<Frame::Grid, Frame::Distorted>,
      IdentityForComposition<Frame::Distorted, Frame::Inertial>,
      detail::produce_all_maps<Frame::Grid, Frame::Distorted, ShapeMap>,
      detail::produce_all_maps<Frame::Grid, Frame::Inertial, ShapeMap>,
      detail::produce_all_maps<Frame::Grid, Frame::Inertial, ShapeMap,
                               ExpansionMap, RotationMap>,
      detail::produce_all_maps<Frame::Grid, Frame::Inertial, ShapeMap,
                               ExpansionMap>,
      detail::produce_all_maps<Frame::Grid, Frame::Inertial, ShapeMap,
                               RotationMap>,
      detail::produce_all_maps<Frame::Distorted, Frame::Inertial, ExpansionMap,
                               RotationMap>,
      detail::produce_all_maps<Frame::Distorted, Frame::Inertial, ExpansionMap>,
      detail::produce_all_maps<Frame::Distorted, Frame::Inertial, RotationMap>>;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  struct ExpansionOptions {
    using type = Options::Auto<ExpansionOptions, Options::AutoLabel::None>;
    static std::string name() { return "Expansion"; }
    static constexpr Options::String help = {
        "Options for a time-dependent expansion map that settles to constant. "
        "Specify 'None' to not use this map."};

    struct InitialValues {
      using type = std::array<double, 3>;
      static constexpr Options::String help = {
          "Initial value and 1st and second derivs of expansion."};
    };
    struct InitialValuesOuterBoundary {
      using type = std::array<double, 3>;
      static constexpr Options::String help = {
          "Initial value and 1st and second derivs of expansion."};
    };
    struct OuterBoundaryRadius {
      using type = double;
      static constexpr Options::String help = {
          "Radius of outer boundary used in expansion map."};
    };
    struct MatchTime {
      using type = double;
      static constexpr Options::String help = {"The match time."};
    };
    struct DecayTime {
      using type = double;
      static constexpr Options::String help = {"The decay time."};
    };

    using options = tmpl::list<InitialValues, InitialValuesOuterBoundary,
                               OuterBoundaryRadius, MatchTime, DecayTime>;

    std::array<double, 3> initial_values{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    std::array<double, 3> initial_values_outer_boundary{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    double outer_boundary_radius{std::numeric_limits<double>::signaling_NaN()};
    double match_time{std::numeric_limits<double>::signaling_NaN()};
    double decay_time{std::numeric_limits<double>::signaling_NaN()};
  };

  struct RotationOptions {
    using type = Options::Auto<RotationOptions, Options::AutoLabel::None>;
    static std::string name() { return "Rotation"; }
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map that settles to constant. "
        "Specify 'None' to not use this map."};
    struct InitialQuaternionValues {
      using type = std::array<double, 4>;
      static constexpr Options::String help = {
          "The initial quaternion values."};
    };
    struct InitialQuaternionFirstDerivatives {
      using type = std::array<double, 4>;
      static constexpr Options::String help = {
          "The initial quaternion first derivatives."};
    };
    struct InitialQuaternionSecondDerivatives {
      using type = std::array<double, 4>;
      static constexpr Options::String help = {
          "The initial quaternion second derivatives."};
    };
    struct MatchTime {
      using type = double;
      static constexpr Options::String help = {"The match time."};
    };
    struct DecayTime {
      using type = double;
      static constexpr Options::String help = {"The decay time."};
    };

    using options =
        tmpl::list<InitialQuaternionValues, InitialQuaternionFirstDerivatives,
                   InitialQuaternionSecondDerivatives, MatchTime, DecayTime>;

    std::array<double, 4> initial_quaternion_values{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    std::array<double, 4> initial_quaternion_first_derivatives{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    std::array<double, 4> initial_quaternion_second_derivatives{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    double match_time{std::numeric_limits<double>::signaling_NaN()};
    double decay_time{std::numeric_limits<double>::signaling_NaN()};
  };

  struct ShapeMapOptions {
    using type = ShapeMapOptions;
    static std::string name() { return "ShapeMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent shape map in the inner-most shell of the "
        "domain."};

    struct LMax {
      using type = size_t;
      static constexpr Options::String help = {
          "Initial LMax for the shape map."};
    };

    struct InitialValues {
      using type =
          Options::Auto<std::variant<KerrSchildFromBoyerLindquist>, Spherical>;
      static constexpr Options::String help = {
          "Initial Ylm coefficients for the shape map. Specify 'Spherical' for "
          "all coefficients to be initialized to zero."};
    };

    using options = tmpl::list<LMax, InitialValues>;

    size_t l_max{};
    std::optional<std::variant<KerrSchildFromBoyerLindquist>> initial_values{};
  };

  using options = tmpl::list<InitialTime, ShapeMapOptions, ExpansionOptions,
                             RotationOptions>;
  static constexpr Options::String help{
      "The options for all the hard-coded time dependent maps in the Sphere "
      "domain."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(double initial_time,
                          const ShapeMapOptions& shape_map_options,
                          std::optional<ExpansionOptions> expansion_map_options,
                          std::optional<RotationOptions> rotation_options);

  /*!
   * \brief Create the function of time map using the options that were
   * provided to this class.
   *
   * Currently, this will add:
   *
   * - Size: `PiecewisePolynomial<3>`
   * - Shape: `PiecewisePolynomial<2>`
   */
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
  create_functions_of_time(double inner_radius,
                           const std::unordered_map<std::string, double>&
                               initial_expiration_times) const;

  /*!
   * \brief Construct the actual maps that will be used.
   *
   * Currently, this constructs a:
   *
   * - Shape: `Shape` (with a size function of time)
   */
  void build_maps(const std::array<double, 3>& center, double inner_radius,
                  double outer_radius);

  /*!
   * \brief This will construct the map from `Frame::Distorted` to
   * `Frame::Inertial`.
   *
   * If the argument `include_distorted_map` is true, then this will be an
   * identity map. If it is false, then this returns `nullptr`.
   */
  MapType<Frame::Distorted, Frame::Inertial> distorted_to_inertial_map(
      bool include_distorted_map) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to
   * `Frame::Distorted`.
   *
   * If the argument `include_distorted_map` is true, then this will add a
   * `Shape` map (with a size function of time). If it is false, then this
   * returns `nullptr`.
   */
  MapType<Frame::Grid, Frame::Distorted> grid_to_distorted_map(
      bool include_distorted_map) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to `Frame::Inertial`.
   *
   * If the argument `include_distorted_map` is true, then this map will have a
   * `Shape` map (with a size function of time). If it is false, then there will
   * only be an identity map.
   */
  MapType<Frame::Grid, Frame::Inertial> grid_to_inertial_map(
      bool include_distorted_map) const;

  inline static const std::string size_name{"Size"};
  inline static const std::string shape_name{"Shape"};
  inline static const std::string expansion_name{"Expansion"};
  inline static const std::string expansion_outer_boundary_name{
      "ExpansionOuterBoundary"};
  inline static const std::string rotation_name{"Rotation"};

 private:
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  size_t initial_l_max_{};
  ShapeMap shape_map_{};
  std::optional<std::variant<KerrSchildFromBoyerLindquist>>
      initial_shape_values_{};

  std::optional<ExpansionOptions> expansion_map_options_{};
  std::optional<RotationOptions> rotation_options_{};
  std::optional<ExpansionMap> expansion_map_{};
  std::optional<RotationMap> rotation_map_{};
};
}  // namespace domain::creators::sphere
