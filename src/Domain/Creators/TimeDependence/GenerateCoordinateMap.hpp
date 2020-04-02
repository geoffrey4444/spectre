// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
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
namespace detail {
template <typename Map>
struct get_maps {
  using type = tmpl::list<Map>;
};
template <typename SourceFrame, typename TargetFrame, typename... Maps>
struct get_maps<domain::CoordinateMap<SourceFrame, TargetFrame, Maps...>>{
  using type = tmpl::list<Maps...>;
};
template <typename MapsList>
struct generate_final_coordinate_map;
template <typename... Maps>
struct generate_final_coordinate_map<tmpl::list<Maps...>> {
  using type = domain::CoordinateMap<Frame::Grid, Frame::Inertial, Maps...>;
};
template <typename CoordinateMapsList>
struct generate_coordinate_map;
template <typename... CoordinateMaps>
struct generate_coordinate_map<tmpl::list<CoordinateMaps...>> {
  using type = typename generate_final_coordinate_map<tmpl::flatten<
      tmpl::list<typename get_maps<CoordinateMaps>::type...>>>::type;
};
template <typename MapsList>
using generate_coordinate_map_t =
    typename generate_coordinate_map<MapsList>::type;
}  // namespace detail
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
