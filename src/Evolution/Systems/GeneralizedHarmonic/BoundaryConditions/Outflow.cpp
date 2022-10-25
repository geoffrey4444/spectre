// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Outflow.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
template <size_t Dim>
Outflow<Dim>::Outflow(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow<Dim>::get_clone() const {
  return std::make_unique<Outflow>(*this);
}

template <size_t Dim>
void Outflow<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
std::optional<std::string> Outflow<Dim>::dg_outflow(
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
    /*outward_directed_normal_vector*/,

    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift) {
  const auto char_speeds = characteristic_speeds(
      gamma_1, lapse, shift, outward_directed_normal_covector);
  Scalar<DataVector> normal_dot_mesh_velocity;
  if (face_mesh_velocity.has_value()) {
    normal_dot_mesh_velocity = dot_product(outward_directed_normal_covector,
                                           face_mesh_velocity.value());
  }
  double min_speed = std::numeric_limits<double>::signaling_NaN();
  for (size_t i = 0; i < char_speeds.size(); ++i) {
    if (face_mesh_velocity.has_value()) {
      min_speed = min(gsl::at(char_speeds, i) - get(normal_dot_mesh_velocity));
    } else {
      min_speed = min(gsl::at(char_speeds, i));
    }
    if (min_speed < -1.e-12) {
      return {MakeString{}
              << "Outflow boundary condition violated with speed index " << i
              << " ingoing: " << min_speed
              << "\n speed: " << gsl::at(char_speeds, i)
              << "\nn_i: " << outward_directed_normal_covector
              << "\n"
                 "See GeneralizedHarmonic::characteristic_speeds for the "
                 "index ordering of characteristic speeds\n"};
    }
  }
  return std::nullopt;
}

// NOLINTNEXTLINE
template <size_t Dim>
PUP::able::PUP_ID Outflow<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class Outflow<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::BoundaryConditions
