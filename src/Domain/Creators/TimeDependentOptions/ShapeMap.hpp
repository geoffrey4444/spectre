// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
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

  KerrSchildFromBoyerLindquist();
  KerrSchildFromBoyerLindquist(double mass_in, std::array<double, 3> spin_in);

  double mass{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> spin{std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN()};
};

/// Label for shape map options
struct Spherical {};

struct YlmsFromFile {
  struct H5Filename {
    using type = std::string;
    static constexpr Options::String help =
        "Path to the data file containing the ylm coefficients and their "
        "derivatives.";
  };

  struct SubfileNames {
    using type = std::vector<std::string>;
    static constexpr Options::String help =
        "Subfile names for the different order derivatives of the ylm "
        "coefficients. You must specify the subfile name for the ylm "
        "coefficients themselves, and can optionally specify the subfile name "
        "for the first and second time derivatives as well, in that order. If "
        "you don't specify a derivative subfile, those coefficients will be "
        "defaulted to zero.";
    static size_t lower_bound_on_size() { return 1; }
    static size_t upper_bound_on_size() { return 3; }
  };

  struct MatchTime {
    using type = double;
    static constexpr Options::String help =
        "Time in the H5File to get the coefficients at. Will likely be the "
        "same as the initial time";
  };

  struct MatchTimeEpsilon {
    using type = Options::Auto<double>;
    static constexpr Options::String help =
        "Look for times in the H5File within this epsilon of the match time. "
        "This is to avoid having to know the exact time to all digits. Default "
        "is 1e-12.";
  };

  struct SetL1CoefsToZero {
    using type = bool;
    static constexpr Options::String help =
        "Whether to set the L=1 coefs to zero or not. This may be desirable "
        "because L=1 is degenerate with a translation of the BH.";
  };

  struct CheckFrame {
    using type = bool;
    static constexpr Options::String help =
        "Whether to check if the frame of the Strahlkorper in the file matches "
        "the Distorted frame.";
  };

  using options = tmpl::list<H5Filename, SubfileNames, MatchTime,
                             MatchTimeEpsilon, SetL1CoefsToZero, CheckFrame>;

  static constexpr Options::String help = {
      "Read the Y_lm coefficients of a Strahlkorper from file and use those to "
      "initialize the coefficients of a shape map."};
  YlmsFromFile();
  YlmsFromFile(std::string h5_filename_in,
               std::vector<std::string> subfile_names_in, double match_time_in,
               std::optional<double> match_time_epsilon_in,
               bool set_l1_coefs_to_zero_in, bool check_frame_in = true);

  std::string h5_filename;
  std::vector<std::string> subfile_names;
  double match_time{};
  std::optional<double> match_time_epsilon;
  bool set_l1_coefs_to_zero{};
  bool check_frame{true};
};

struct YlmsFromSpEC {
  struct DatFilename {
    using type = std::string;
    static constexpr Options::String help =
        "Name of the SpEC dat file holding the coefficients. Note that this "
        "isn't a Dat file within an H5 file. This must be an actual '.dat' "
        "file on disk.";
  };

  struct MatchTime {
    using type = double;
    static constexpr Options::String help =
        "Time in the H5File to get the coefficients at. Will likely be the "
        "same as the initial time";
  };

  struct MatchTimeEpsilon {
    using type = Options::Auto<double>;
    static constexpr Options::String help =
        "Look for times in the H5File within this epsilon of the match time. "
        "This is to avoid having to know the exact time to all digits. Default "
        "is 1e-12.";
  };

  struct SetL1CoefsToZero {
    using type = bool;
    static constexpr Options::String help =
        "Whether to set the L=1 coefs to zero or not. This may be desirable "
        "because L=1 is degenerate with a translation of the BH.";
  };

  using options =
      tmpl::list<DatFilename, MatchTime, MatchTimeEpsilon, SetL1CoefsToZero>;

  static constexpr Options::String help = {
      "Read the Y_lm coefficients of a Strahlkorper from file and use those to "
      "initialize the coefficients of a shape map."};
  YlmsFromSpEC();
  YlmsFromSpEC(std::string dat_filename_in, double match_time_in,
               std::optional<double> match_time_epsilon_in,
               bool set_l1_coefs_to_zero_in);

  std::string dat_filename;
  double match_time{};
  std::optional<double> match_time_epsilon;
  bool set_l1_coefs_to_zero{};
};

namespace detail {
struct TransitionEndsAtCube {
  using type = bool;
  static constexpr Options::String help = {
      "If 'true', the shape map transition function will be 0 at the cubical "
      "boundary around the object. If 'false' the transition function will "
      "be 0 at the outer radius of the inner sphere around the object"};
};
}  // namespace detail

/*!
 * \brief Specialized version of `FromVolumeFile` for the shape map
 *
 * \details This is needed because the regular `FromVolumeFile` doesn't have
 * options for domain settings like `TransitionEndsAtCube` or `LMax`.
 */
template <ObjectLabel Object>
struct FromVolumeFileShapeSize : public FromVolumeFile {
 public:
  struct CoefficientTruncationLimit {
    using type = double;
    static constexpr Options::String help = {
        "Coefficients below this absolute value will be truncated from the "
        "Shape map. Set to 0.0 to disable truncation."};
    static constexpr type default_value = 0.0;
  };
  struct LMax {
    using type = Options::Auto<size_t>;
    static constexpr Options::String help = {
        "LMax used for the number of spherical harmonic coefficients of the "
        "distortion map. If set to 'Auto', will use the LMax from the shape "
        "function of time in the volume file."};
  };
  using options = tmpl::push_front<FromVolumeFile::options, LMax,
                                   CoefficientTruncationLimit,
                                   detail::TransitionEndsAtCube>;

  FromVolumeFileShapeSize() = default;
  FromVolumeFileShapeSize(const std::optional<size_t>& l_max_in,
                          double coefficient_truncation_limit_in,
                          bool transition_ends_at_cube_in,
                          std::string h5_filename, std::string subfile_name,
                          const Options::Context& context = {});

  size_t l_max{};
  double coefficient_truncation_limit{0.0};
  bool transition_ends_at_cube{};

 private:
  std::string h5_filename_;
  std::string subfile_name_;
};

/*!
 * \brief Class to be used as an option for initializing shape map coefficients.
 *
 * \details This class can also be used as an option tag with the \p type type
 * alias, `name()` function, and \p help string.
 *
 * \tparam IncludeTransitionEndsAtCube This is mainly added for the
 * `domain::creators::BinaryCompactObject` domain.
 * \tparam Object Which object that this shape map represents. Use
 * `domain::ObjectLabel::None` if there is only a single object in your
 * simulation.
 */
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
struct ShapeMapOptions {
  struct CoefficientTruncationLimit {
    using type = double;
    static constexpr Options::String help = {
        "Coefficients below this absolute value will be truncated from the "
        "Shape map. Set to 0.0 to disable truncation."};
    static constexpr type default_value = 0.0;
  };
  using type = Options::Auto<
      std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                   FromVolumeFileShapeSize<Object>>,
      Options::AutoLabel::None>;
  static std::string name() { return "ShapeMap" + get_output(Object); }
  static constexpr Options::String help = {
      "Options for a time-dependent distortion (shape) map about the "
      "specified object. Specify 'None' to not use this map."};

  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "LMax used for the number of spherical harmonic coefficients of the "
        "distortion map."};
  };

  struct InitialValues {
    using type = Options::Auto<
        std::variant<KerrSchildFromBoyerLindquist, YlmsFromFile, YlmsFromSpEC>,
        Spherical>;
    static constexpr Options::String help = {
        "Initial Ylm coefficients for the shape map. Specify 'Spherical' for "
        "all coefficients to be initialized to zero."};
  };

  struct SizeInitialValues {
    struct Value {
      using type = Options::Auto<double>;
      static std::string name() { return "Value"; }
      static constexpr Options::String help = {
          "Initial value of the 00 coefficient. Specify 'Auto' to use the 00 "
          "coefficient specified in the 'InitialValues' option."};
    };
    struct Derivatives {
      using type = std::array<double, 2>;
      static std::string name() { return "Derivatives"; }
      static constexpr Options::String help = {
          "First two time derivatives of the 00 coefficient."};
    };
    struct ValueAndDerivatives {
      using options = tmpl::list<Value, Derivatives>;
      static constexpr Options::String help = {
          "Initial value and two derivatives of the 00 coefficient."};
      ValueAndDerivatives() = default;
      ValueAndDerivatives(std::optional<double> value_in,
                          std::array<double, 2> derivatives_in)
          : value(value_in), derivatives(std::move(derivatives_in)) {}

      std::optional<double> value;
      std::array<double, 2> derivatives{};
    };

    using type =
        Options::Auto<std::variant<std::array<double, 3>, ValueAndDerivatives>>;
    static constexpr Options::String help = {
        "Initial value and two derivatives of the 00 coefficient. Specify "
        "'Auto' to use the 00 coefficient specified in the 'InitialValues' "
        "option. If you specify 'Auto', the deformed sphere will match the "
        "'InitialValues' surface exactly, and the original radius will only "
        "set the radius of the sphere in the grid frame (before deformation)."};
  };

  using common_options = tmpl::list<LMax, InitialValues, SizeInitialValues,
                                    CoefficientTruncationLimit>;

  using options = tmpl::conditional_t<
      IncludeTransitionEndsAtCube,
      tmpl::push_back<common_options, detail::TransitionEndsAtCube>,
      common_options>;
  ShapeMapOptions() = default;
  ShapeMapOptions(
      size_t l_max_in,
      std::optional<std::variant<KerrSchildFromBoyerLindquist, YlmsFromFile,
                                 YlmsFromSpEC>>
          initial_values_in,
      std::optional<
          std::variant<std::array<double, 3>,
                       typename SizeInitialValues::ValueAndDerivatives>>
          initial_size_values_in = std::nullopt,
      double coefficient_truncation_limit_in = 0.0,
      bool transition_ends_at_cube_in = false)
      : l_max(l_max_in),
        initial_values(std::move(initial_values_in)),
        coefficient_truncation_limit(coefficient_truncation_limit_in),
        transition_ends_at_cube(transition_ends_at_cube_in) {
    if (initial_size_values_in.has_value()) {
      if (std::holds_alternative<std::array<double, 3>>(
              initial_size_values_in.value())) {
        initial_size_values =
            std::get<std::array<double, 3>>(initial_size_values_in.value());
      } else {
        const auto& value_and_derivatives =
            std::get<typename SizeInitialValues::ValueAndDerivatives>(
                initial_size_values_in.value());
        initial_size_value = value_and_derivatives.value;
        initial_size_derivatives = value_and_derivatives.derivatives;
      }
    }
  }
  ShapeMapOptions(size_t l_max_in,
                  std::optional<std::variant<KerrSchildFromBoyerLindquist,
                                             YlmsFromFile, YlmsFromSpEC>>
                      initial_values_in,
                  std::array<double, 3> initial_size_values_in,
                  double coefficient_truncation_limit_in = 0.0,
                  bool transition_ends_at_cube_in = false)
      : ShapeMapOptions(
            l_max_in, std::move(initial_values_in),
            std::optional<
                std::variant<std::array<double, 3>,
                             typename SizeInitialValues::ValueAndDerivatives>>{
                std::move(initial_size_values_in)},
            coefficient_truncation_limit_in, transition_ends_at_cube_in) {}
  ShapeMapOptions(size_t l_max_in,
                  std::optional<std::variant<KerrSchildFromBoyerLindquist,
                                             YlmsFromFile, YlmsFromSpEC>>
                      initial_values_in,
                  std::optional<double> initial_size_value_in,
                  const std::array<double, 2>& initial_size_derivatives_in,
                  double coefficient_truncation_limit_in = 0.0,
                  bool transition_ends_at_cube_in = false)
      : l_max(l_max_in),
        initial_values(std::move(initial_values_in)),
        initial_size_value(initial_size_value_in),
        initial_size_derivatives(initial_size_derivatives_in),
        coefficient_truncation_limit(coefficient_truncation_limit_in),
        transition_ends_at_cube(transition_ends_at_cube_in) {}

  size_t l_max{};
  std::optional<
      std::variant<KerrSchildFromBoyerLindquist, YlmsFromFile, YlmsFromSpEC>>
      initial_values;
  std::optional<std::array<double, 3>> initial_size_values;
  std::optional<double> initial_size_value;
  std::optional<std::array<double, 2>> initial_size_derivatives;
  double coefficient_truncation_limit{0.0};
  bool transition_ends_at_cube{false};
};

/*!
 * \brief Helper function to get LMax from the different variants that the shape
 * map options could be.
 */
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
size_t l_max_from_shape_options(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options);

/*!
 * \brief Helper function to get the coefficient truncation limit from the
 * different variants that the shape map options could be.
 */
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
double coefficient_truncation_limit_from_shape_options(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options);

/*!
 * \brief Helper function to get whether the shape map transition function ends
 * at the cube from the different variants that the shape map options could be.
 */
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
bool transition_ends_at_cube_from_shape_options(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options);

/*!
 * \brief Helper function that takes the variant of the shape map options, and
 * returns the fully constructed shape and size functions of time.
 *
 * \details Even if the functions of time are read from a file, they will have a
 * new \p initial_time, \p shape_expiration_time, and \p size_expiration_time.
 * The \p deformed_radius is only used for the non-volume file variants.
 */
template <bool IncludeTransitionEndsAtCube, domain::ObjectLabel Object>
FunctionsOfTimeMap get_shape_and_size(
    const std::variant<ShapeMapOptions<IncludeTransitionEndsAtCube, Object>,
                       FromVolumeFileShapeSize<Object>>& shape_map_options,
    double initial_time, double shape_expiration_time,
    double size_expiration_time, double deformed_radius);
}  // namespace domain::creators::time_dependent_options
