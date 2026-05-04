// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/TensorYlmTransforms.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.tpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

namespace ylm::TensorYlm {

namespace {

template <typename Tag>
void apply_cartesian_to_tensor_ylm_matrix(
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        result,
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        temp_storage,
    const SimpleSparseMatrix& matrix, const size_t radial_extents) {
  constexpr size_t num_independent_components = Tag::type::structure::size();
  ASSERT(result->number_of_grid_points() * num_independent_components <=
             temp_storage->size(),
         "Insufficient temporary storage for tensor-Ylm transform: need "
             << result->number_of_grid_points() * num_independent_components
             << " doubles but have " << temp_storage->size() << ".");

  Variables<tmpl::list<Tag>> transformed_tensor(
      temp_storage->data(),
      result->number_of_grid_points() * num_independent_components);
  for (auto& component : get<Tag>(transformed_tensor)) {
    component = 0.0;
  }

  const gsl::span<double> src(
      get<Tag>(*result)[0].data(),
      num_independent_components * result->number_of_grid_points());
  gsl::span<double> dest(
      get<Tag>(transformed_tensor)[0].data(),
      num_independent_components * transformed_tensor.number_of_grid_points());
  for (size_t offset = 0; offset < radial_extents; ++offset) {
    matrix.increment_multiply_on_right(make_not_null(&dest), offset,
                                       radial_extents, src, offset,
                                       radial_extents);
  }
  get<Tag>(*result) = get<Tag>(transformed_tensor);
}

}  // namespace

void gh_variables_to_tensor_ylm_coefficients(
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        gh_spatial_tensor_ylm_coefficients,
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        temp_storage,
    const Variables<filter_detail::gh_spacetime_vars_list>& gh_vars,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const SimpleSparseMatrix& cart_to_sphere_matrix_i,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ii,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ij,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ijj,
    const Spherepack& spherepack, const size_t radial_extents) {
  const size_t physical_size = radial_extents * spherepack.physical_size();
  const size_t spectral_size = radial_extents * spherepack.spectral_size();
  ASSERT(gh_vars.number_of_grid_points() == physical_size,
         "Expected GH variables to have "
             << physical_size << " grid points, but got "
             << gh_vars.number_of_grid_points() << ".");
  ASSERT(gh_spatial_tensor_ylm_coefficients->number_of_grid_points() ==
             spectral_size,
         "Expected output tensor-Ylm coefficients to have "
             << spectral_size
             << " grid points, "
                "but got "
             << gh_spatial_tensor_ylm_coefficients->number_of_grid_points()
             << ".");
  ASSERT(temp_storage->number_of_grid_points() >= physical_size,
         "Expected temporary storage to hold at least "
             << physical_size
             << " grid points, but "
                "got "
             << temp_storage->number_of_grid_points() << ".");
  ASSERT(gh_spatial_tensor_ylm_coefficients->data() != temp_storage->data(),
         "The output tensor-Ylm coefficients must not alias temp_storage.");

  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      gh_spatial_inertial_vars(
          gh_spatial_tensor_ylm_coefficients->data(),
          physical_size *
              Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>::
                  number_of_independent_components);
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>
      gh_spatial_grid_vars(
          temp_storage->data(),
          physical_size *
              Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>::
                  number_of_independent_components);

  filter_detail::break_spacetime_vars_into_spatial_pieces(
      make_not_null(&gh_spatial_inertial_vars), gh_vars);
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&gh_spatial_grid_vars),
                                    gh_spatial_inertial_vars,
                                    jac_inertial_to_grid);
  filter_detail::nodal_to_modal_ylm(gh_spatial_tensor_ylm_coefficients,
                                    gh_spatial_grid_vars, spherepack,
                                    radial_extents);

  tmpl::for_each<filter_detail::gh_spatial_vars_list<
      Frame::Grid>>([gh_spatial_tensor_ylm_coefficients, temp_storage,
                     radial_extents, &cart_to_sphere_matrix_i,
                     &cart_to_sphere_matrix_ii, &cart_to_sphere_matrix_ij,
                     &cart_to_sphere_matrix_ijj]<class Tag>(
                        const tmpl::type_<Tag> /*meta*/) {
    if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                 Symmetry<1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, temp_storage,
          cart_to_sphere_matrix_i, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<1, 1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, temp_storage,
          cart_to_sphere_matrix_ii, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<2, 1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, temp_storage,
          cart_to_sphere_matrix_ij, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<2, 1, 1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, temp_storage,
          cart_to_sphere_matrix_ijj, radial_extents);
    }
  });
}

}  // namespace ylm::TensorYlm
