// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
/// Options for sampling characteristic speeds on rescaled apparent horizons.
struct RescaledSurfaceCharSpeedOptions {
  struct ExcisionSphere {
    using type = std::string;
    static constexpr Options::String help =
        "Name of the domain excision sphere inside the apparent horizon.";
  };
  struct NumberOfSurfaces {
    using type = size_t;
    static constexpr Options::String help =
        "Number of surfaces, including the innermost surface and horizon. "
        "Use 10 for the standard diagnostic.";
  };
  struct RelativeExcisionMargin {
    using type = double;
    static constexpr Options::String help =
        "Fractional outward displacement from the excision surface. "
        "Use 1.e-7 for the standard diagnostic.";
  };
  using options =
      tmpl::list<ExcisionSphere, NumberOfSurfaces, RelativeExcisionMargin>;
  static constexpr Options::String help =
      "Observe characteristic speeds on rescaled apparent horizons.";

  RescaledSurfaceCharSpeedOptions() = default;
  explicit RescaledSurfaceCharSpeedOptions(
      std::string excision_sphere_in, size_t number_of_surfaces_in = 10,
      double relative_excision_margin_in = 1.e-7,
      const Options::Context& context = {});

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  std::string excision_sphere;
  size_t number_of_surfaces{10};
  double relative_excision_margin{1.e-7};
};

bool operator==(const RescaledSurfaceCharSpeedOptions& lhs,
                const RescaledSurfaceCharSpeedOptions& rhs);
bool operator!=(const RescaledSurfaceCharSpeedOptions& lhs,
                const RescaledSurfaceCharSpeedOptions& rhs);

/// Options for finding an apparent horizon.
template <typename Fr>
struct HorizonOptions {
 private:
  struct All {};

 public:
  struct Criteria {
    static constexpr Options::String help = {
        "List of criteria for adapting the horizon resolution"};
    using type = std::vector<std::unique_ptr<ah::Criterion>>;
  };
  /// See Strahlkorper for suboptions.
  struct InitialGuess {
    static constexpr Options::String help = {"Initial guess"};
    using type = ylm::Strahlkorper<Fr>;
  };
  /// See ::FastFlow for suboptions.
  struct FastFlow {
    static constexpr Options::String help = {"FastFlow options"};
    using type = ::FastFlow;
  };
  struct Verbosity {
    static constexpr Options::String help = {"Verbosity"};
    using type = ::Verbosity;
  };
  struct MaxComputeCoordsRetries {
    static constexpr Options::String help = {
        "Number of times to retry computing the coordinates of the horizon for "
        "each iteration. For the zeroth iteration, increases the 00 component "
        "by 50%. For subsequent iterations, two previous surfaces are averaged "
        "and that new surface is used."};
    using type = size_t;
  };
  struct BlocksForHorizonFind {
    static constexpr Options::String help = {
        "Volume data will be sent to the horizon finder from these block group "
        "names. Set to 'All' to send volume data from the entire domain."};
    using type = Options::Auto<std::vector<std::string>, All>;
  };
  struct RescaledSurfaceCharSpeeds {
    using type = Options::Auto<RescaledSurfaceCharSpeedOptions,
                               Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Optional characteristic-speed observations on rescaled horizons. "
        "Omit or use None to disable.";
  };
  using common_options =
      tmpl::list<Criteria, InitialGuess, FastFlow, Verbosity,
                 MaxComputeCoordsRetries, BlocksForHorizonFind>;
  using options = tmpl::push_back<common_options, RescaledSurfaceCharSpeeds>;
  static constexpr Options::String help = {
      "Provide an initial guess for the apparent horizon surface\n"
      "(Strahlkorper) and apparent-horizon-finding-algorithm (FastFlow)\n"
      "options."};

  HorizonOptions(
      std::vector<std::unique_ptr<ah::Criterion>> criteria_in,
      ylm::Strahlkorper<Fr> initial_guess_in, ::FastFlow fast_flow_in,
      ::Verbosity verbosity_in, size_t max_compute_coords_retries_in,
      std::optional<std::vector<std::string>> blocks_for_horizon_find_in,
      std::optional<RescaledSurfaceCharSpeedOptions>
          rescaled_surface_char_speeds_in = std::nullopt,
      const Options::Context& context = {});

  HorizonOptions() = default;
  HorizonOptions(const HorizonOptions& /*rhs*/) = delete;
  HorizonOptions& operator=(const HorizonOptions& /*rhs*/) = delete;
  HorizonOptions(HorizonOptions&& /*rhs*/) = default;
  HorizonOptions& operator=(HorizonOptions&& /*rhs*/) = default;
  ~HorizonOptions() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  std::vector<std::unique_ptr<ah::Criterion>> criteria;
  ylm::Strahlkorper<Fr> initial_guess{};
  ::FastFlow fast_flow;
  ::Verbosity verbosity{::Verbosity::Quiet};
  size_t max_compute_coords_retries{};
  std::optional<std::vector<std::string>> blocks_for_horizon_find;
  std::optional<RescaledSurfaceCharSpeedOptions> rescaled_surface_char_speeds;
};

template <typename Fr>
bool operator==(const HorizonOptions<Fr>& lhs, const HorizonOptions<Fr>& rhs);
template <typename Fr>
bool operator!=(const HorizonOptions<Fr>& lhs, const HorizonOptions<Fr>& rhs);

namespace OptionTags {
struct ApparentHorizonGroup {
  static constexpr Options::String help{"Options for apparent horizon finders"};
  static std::string name() { return "ApparentHorizons"; }
};

template <typename HorizonMetavars>
struct ApparentHorizonOptions {
  using type = HorizonOptions<typename HorizonMetavars::frame>;
  static constexpr Options::String help{
      "Options for interpolation onto apparent horizon."};
  static std::string name() { return pretty_type::name<HorizonMetavars>(); }
  using group = ApparentHorizonGroup;
};

/// \ingroup OptionTagsGroup
/// Maximum L used both for adaptive horizon resolution and output padding.
struct LMax {
  using type = size_t;
  static constexpr Options::String help = {
      "Maximum L for horizon resolution and output. Adaptive criteria clamp "
      "the surface to this L, and output at smaller L is zero padded to "
      "match this maximum L."};
  using group = ApparentHorizonGroup;
};
}  // namespace OptionTags
}  // namespace ah

namespace Options {
/// Preserve input files that predate the optional rescaled-surface diagnostic.
template <typename Fr>
struct create_from_yaml<ah::HorizonOptions<Fr>> {
  template <typename Metavariables>
  static ah::HorizonOptions<Fr> create(const Option& option) {
    using HorizonOptions = ah::HorizonOptions<Fr>;
    const auto parse_options = [&option]<typename... Tags>(
                                   tmpl::list<Tags...> /*meta*/) {
      using OptionList = tmpl::list<Tags...>;
      Parser<OptionList> parser{HorizonOptions::help};
      parser.parse(option);
      return parser.template apply<OptionList,
                                   Metavariables>([&option](auto&&... args) {
        if constexpr (tmpl::list_contains_v<
                          OptionList,
                          typename HorizonOptions::RescaledSurfaceCharSpeeds>) {
          return HorizonOptions(std::forward<decltype(args)>(args)...,
                                option.context());
        } else {
          return HorizonOptions(std::forward<decltype(args)>(args)...,
                                std::nullopt, option.context());
        }
      });
    };
    if (option.node().IsMap() and option.node()["RescaledSurfaceCharSpeeds"]) {
      return parse_options(typename HorizonOptions::options{});
    }
    return parse_options(typename HorizonOptions::common_options{});
  }
};
}  // namespace Options
