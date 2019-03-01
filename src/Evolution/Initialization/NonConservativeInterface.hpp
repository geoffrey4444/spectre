// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Initialization {

/// \brief Initialize items related to the interfaces between Elements and on
/// external boundaries, for non-conservative systems
///
/// DataBox changes:
/// - Adds:
///   * `face_tags<Tags::InternalDirections<Dim>>`
///   * `face_tags<Tags::BoundaryDirectionsInterior<Dim>>`
///   * `face_tags<Tags::BoundaryDirectionsExterior<Dim>>`
///
/// - For face_tags:
///   * `Tags::InterfaceComputeItem<Directions, Tags::Direction<Dim>>`
///   * `Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<Dim>>`
///   * `Tags::Slice<Directions, typename System::variables_tag>`
///   * `Tags::InterfaceComputeItem<Directions,
///                                Tags::UnnormalizedFaceNormal<Dim>>`
///   * Tags::InterfaceComputeItem<Directions,
///                                typename System::template magnitude_tag<
///                                    Tags::UnnormalizedFaceNormal<Dim>>>
///   * `Tags::InterfaceComputeItem<
///         Directions, Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>`
/// - Removes: nothing
/// - Modifies: nothing
template <typename System>
struct InterfaceForNonConservativeSystem {
  static constexpr size_t dim = System::volume_dim;
  using frame = Frame::Inertial;
  using simple_tags =
      db::AddSimpleTags<Tags::Interface<Tags::BoundaryDirectionsExterior<dim>,
                                        typename System::variables_tag>>;

  template <typename Directions>
  using face_tags = tmpl::list<
      Directions, Tags::InterfaceComputeItem<Directions, Tags::Direction<dim>>,
      Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<dim>>,
      Tags::Slice<Directions, typename System::variables_tag>,
      Tags::InterfaceComputeItem<Directions,
                                 gr::Tags::SpatialMetricCompute<dim, frame,
                                                                DataVector>>,
      Tags::InterfaceComputeItem<Directions,
                                 gr::Tags::InverseSpatialMetricCompute<dim,
                                     frame, DataVector>>,
      Tags::InterfaceComputeItem<Directions,
                                 gr::Tags::LapseCompute<dim, frame,
                                                        DataVector>>,
      Tags::InterfaceComputeItem<Directions,
                                 gr::Tags::ShiftCompute<dim, frame,
                                                        DataVector>>,
/*      Tags::InterfaceComputeItem<Directions,
                                 GeneralizedHarmonic::Tags::ConstraintGamma0>,
      Tags::InterfaceComputeItem<Directions,
                                 GeneralizedHarmonic::Tags::ConstraintGamma1>,
      Tags::InterfaceComputeItem<Directions,
                                 GeneralizedHarmonic::Tags::ConstraintGamma2>,*/
      Tags::InterfaceComputeItem<Directions, Tags::UnnormalizedFaceNormal<dim>>,
      Tags::InterfaceComputeItem<Directions,
                                 typename System::template magnitude_tag<
                                     Tags::UnnormalizedFaceNormal<dim>>>,
      Tags::InterfaceComputeItem<
          Directions, Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>,
      Tags::InterfaceComputeItem<Directions,
                                 Tags::UnitFaceNormalCompute<dim, frame>>,
      Tags::Interface<Directions, Tags::UnitFaceNormal<dim, frame>>,
      Tags::InterfaceComputeItem<Directions,
                                 Tags::UnitFaceNormalVectorCompute<dim, frame>>,
      Tags::Interface<Directions, Tags::UnitFaceNormalVector<dim, frame>>,
      Tags::InterfaceComputeItem<
          Directions,
          GeneralizedHarmonic::CharacteristicFieldsCompute<dim, frame>>,
      Tags::InterfaceComputeItem<
          Directions,
          GeneralizedHarmonic::CharacteristicSpeedsCompute<dim, frame>>>;

  using ext_tags = tmpl::list<
      Tags::BoundaryDirectionsExterior<dim>,
      Tags::InterfaceComputeItem<
          Tags::BoundaryDirectionsExterior<dim>,
          gr::Tags::SpatialMetricCompute<dim, frame, DataVector>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>,
          gr::Tags::InverseSpatialMetricCompute<dim, frame, DataVector>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>,
                                 Tags::Direction<dim>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>,
                                 Tags::InterfaceMesh<dim>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>,
                                 Tags::BoundaryCoordinates<dim>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>,
                                 Tags::UnnormalizedFaceNormal<dim>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>,
                                 typename System::template magnitude_tag<
                                     Tags::UnnormalizedFaceNormal<dim>>>,
      Tags::Interface<Tags::BoundaryDirectionsExterior<dim>,
                      GeneralizedHarmonic::Tags::ConstraintGamma0>,
      Tags::Interface<Tags::BoundaryDirectionsExterior<dim>,
                      GeneralizedHarmonic::Tags::ConstraintGamma1>,
      Tags::Interface<Tags::BoundaryDirectionsExterior<dim>,
                      GeneralizedHarmonic::Tags::ConstraintGamma2>,
      Tags::Interface<
          Tags::BoundaryDirectionsExterior<dim>,
          Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>,
      Tags::InterfaceComputeItem<
          Tags::BoundaryDirectionsExterior<dim>,
          Tags::UnitFaceNormalCompute<dim, Frame::Inertial>>,
      Tags::Interface<Tags::BoundaryDirectionsExterior<dim>,
                      Tags::UnitFaceNormal<dim, Frame::Inertial>>,
      Tags::InterfaceComputeItem<
          Tags::BoundaryDirectionsExterior<dim>,
          Tags::UnitFaceNormalVectorCompute<dim, Frame::Inertial>>,
      Tags::Interface<Tags::BoundaryDirectionsExterior<dim>,
                      Tags::UnitFaceNormalVector<dim, Frame::Inertial>>,
      Tags::InterfaceComputeItem<
          Tags::BoundaryDirectionsExterior<dim>,
          Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>
          >;

  using compute_tags =
      tmpl::append<face_tags<Tags::InternalDirections<dim>>,
                   face_tags<Tags::BoundaryDirectionsInterior<dim>>, ext_tags>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    const auto& mesh = db::get<Tags::Mesh<dim>>(box);
    std::unordered_map<Direction<dim>,
                       db::item_type<typename System::variables_tag>>
        external_boundary_vars{};

    for (const auto& direction :
         db::get<Tags::Element<dim>>(box).external_boundaries()) {
      external_boundary_vars[direction] =
          db::item_type<typename System::variables_tag>{
              mesh.slice_away(direction.dimension()).number_of_grid_points()};
    }

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(external_boundary_vars));
  }
};

}  // namespace Initialization
