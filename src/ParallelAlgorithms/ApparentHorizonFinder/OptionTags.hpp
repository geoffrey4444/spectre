// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace ah {
/// Selects which elements in the eligible blocks send data to a horizon find.
enum class ElementSendPolicy { Uninitialized, All, PreviousSurfaceNeighbors };

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
        "Block group names eligible to send volume data to the horizon finder. "
        "Set to 'All' to make every block eligible."};
    using type = Options::Auto<std::vector<std::string>, All>;
  };
  struct ElementSendPolicy {
    static constexpr Options::String help = {
        "'All' sends volume data from every element in the eligible blocks. "
        "'PreviousSurfaceNeighbors' sends only from elements intersecting or "
        "neighboring the previous surface. Use 'All' to avoid a deadlock if "
        "the surface moves beyond the filtered elements or the cached previous "
        "surface is stale."};
    using type = ah::ElementSendPolicy;
  };
  using options = tmpl::list<Criteria, InitialGuess, FastFlow, Verbosity,
                             MaxComputeCoordsRetries, BlocksForHorizonFind,
                             ElementSendPolicy>;
  static constexpr Options::String help = {
      "Provide an initial guess for the apparent horizon surface\n"
      "(Strahlkorper) and apparent-horizon-finding-algorithm (FastFlow)\n"
      "options."};

  HorizonOptions(
      std::vector<std::unique_ptr<ah::Criterion>> criteria_in,
      ylm::Strahlkorper<Fr> initial_guess_in, ::FastFlow fast_flow_in,
      ::Verbosity verbosity_in, size_t max_compute_coords_retries_in,
      std::optional<std::vector<std::string>> blocks_for_horizon_find_in,
      ah::ElementSendPolicy element_send_policy_in);

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
  ah::ElementSendPolicy element_send_policy{
      ah::ElementSendPolicy::Uninitialized};
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

template <>
struct Options::create_from_yaml<ah::ElementSendPolicy> {
  template <typename Metavariables>
  static ah::ElementSendPolicy create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
ah::ElementSendPolicy
Options::create_from_yaml<ah::ElementSendPolicy>::create<void>(
    const Options::Option& options);
