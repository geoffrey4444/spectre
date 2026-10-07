// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"

namespace gh {
/*!
 * \brief Experimental radial prolongation/restriction of local GH sources.
 *
 * The evolved radial mesh retains its original extent. Evaluation at32 or40
 * Legendre--Gauss--Lobatto points does not access the high-extent Spectral
 * caches. The modal restriction keeps the original polynomial space.
 *
 * Optional endpoint preservation constrains the source *increment*, for all
 * components, to zero value and logical radial derivative at xi=-1 using the
 * exact unweighted polynomial mass. It does not alter evolved fields or Phi.
 * Actual mapped/tensor-filtered constraint response needs separate checks.
 *
 * This class supplies numerical operators only. It does not replace the GH
 * derivative, damping, boundary, filtering or time-stepping operators.
 */
class RadialSourceProjection {
 public:
  RadialSourceProjection(size_t radial_points, size_t evaluation_points);

  size_t radial_points() const { return radial_points_; }
  size_t evaluation_points() const { return evaluation_points_; }

  const DataVector& evaluation_nodes() const { return evaluation_nodes_; }
  const DataVector& evaluation_weights() const { return evaluation_weights_; }
  const Matrix& prolongation() const { return prolongation_; }
  const Matrix& restriction() const { return restriction_; }
  const Matrix& preserve_inner_endpoint() const { return preserve_inner_; }

  // Component-major layout; within a component radial index varies fastest,
  // followed by the unchanged angular indices, matching Variables on shells.
  DataVector prolong(const DataVector& values, size_t components) const;
  DataVector source_increment(const DataVector& coarse_source,
                              const DataVector& high_source, size_t components,
                              bool preserve_inner) const;

 private:
  static DataVector apply(const Matrix& matrix, const DataVector& values,
                          size_t components);

  size_t radial_points_;
  size_t evaluation_points_;
  DataVector evaluation_nodes_;
  DataVector evaluation_weights_;
  Matrix prolongation_;
  Matrix restriction_;
  Matrix preserve_inner_;
};
}  // namespace gh
