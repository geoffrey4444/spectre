// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/RadialSourceProjectionOptions.hpp"

#include <algorithm>
#include <utility>

#include "Options/ParseError.hpp"

namespace gh {
RadialSourceProjectionOptions::RadialSourceProjectionOptions(
    const size_t points, std::vector<size_t> blocks, const bool preserve_inner,
    const Options::Context& context)
    : evaluation_points(points),
      block_ids(std::move(blocks)),
      preserve_inner_source(preserve_inner) {
  if (points != 32 and points != 40) {
    PARSE_ERROR(context,
                "Experimental source evaluation points must be32 or40.");
  }
  if (block_ids.empty()) {
    PARSE_ERROR(context, "Experimental source projection needs a shell block.");
  }
  std::sort(block_ids.begin(), block_ids.end());
  if (std::adjacent_find(block_ids.begin(), block_ids.end()) !=
      block_ids.end()) {
    PARSE_ERROR(context, "Repeated experimental source-projection block ID.");
  }
}
void RadialSourceProjectionOptions::pup(PUP::er& p) {
  p | evaluation_points;
  p | block_ids;
  p | preserve_inner_source;
}
}  // namespace gh
