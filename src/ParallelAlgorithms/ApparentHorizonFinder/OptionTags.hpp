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
/// Options for finding an apparent horizon.
template <typename Fr>
struct HorizonOptions {
 private:
  struct All {};
  struct Adaptive {};

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
  struct NumberOfPointsForTimeExtrapolation {
    using type = Options::Auto<size_t, Adaptive>;
    static constexpr size_t default_points = 3;
    static constexpr Options::String help = {
        "Use the most recent 2, 3 (default), or 4 horizons to extrapolate "
        "the next initial guess. Adaptive uses the point count with the "
        "smallest measured coefficient error on the last converged horizon. "
        "Ties favor fewer points. Until an error is available, Adaptive uses "
        "as many points as available, up to four. The horizon time derivative "
        "always uses at most three surfaces."};
  };
  using options = tmpl::list<Criteria, InitialGuess, FastFlow, Verbosity,
                             MaxComputeCoordsRetries, BlocksForHorizonFind,
                             NumberOfPointsForTimeExtrapolation>;
  static constexpr Options::String help = {
      "Provide an initial guess for the apparent horizon surface\n"
      "(Strahlkorper) and apparent-horizon-finding-algorithm (FastFlow)\n"
      "options."};

  HorizonOptions(
      std::vector<std::unique_ptr<ah::Criterion>> criteria_in,
      ylm::Strahlkorper<Fr> initial_guess_in, ::FastFlow fast_flow_in,
      ::Verbosity verbosity_in, size_t max_compute_coords_retries_in,
      std::optional<std::vector<std::string>> blocks_for_horizon_find_in,
      std::optional<size_t> number_of_points_for_time_extrapolation_in =
          NumberOfPointsForTimeExtrapolation::default_points,
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
  // An empty value selects adaptive extrapolation.
  std::optional<size_t> number_of_points_for_time_extrapolation{
      NumberOfPointsForTimeExtrapolation::default_points};
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

template <typename Fr>
struct Options::create_from_yaml<ah::HorizonOptions<Fr>> {
  template <typename Metavariables>
  static ah::HorizonOptions<Fr> create(const Options::Option& option) {
    using T = ah::HorizonOptions<Fr>;
    Options::Parser<typename T::options> parser(T::help);
    parser.parse(with_default_extrapolation(option));
    return parser.template apply<typename T::options, Metavariables>(
        [&option](auto... args) {
          return T(std::move(args)..., option.context());
        });
  }

 private:
  // Older input files omit NumberOfPointsForTimeExtrapolation.
  static Options::Option with_default_extrapolation(
      const Options::Option& option);
};
