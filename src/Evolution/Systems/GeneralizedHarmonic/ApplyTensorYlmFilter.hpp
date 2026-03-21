// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace ylm {
class Spherepack;
}  // namespace ylm
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace ylm::TensorYlm {

/// Defines tags and functions used internally in filtering, but
/// tested independently in the unit tests.
namespace filter_detail {

/// Defines tags for internal use in filtering.
namespace Tags {

/// The time-time part of the metric.
template <typename DataType>
struct Metric00 : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The time-time part of the generalized harmonic variable \f$Pi\f$
template <typename DataType>
struct Pi00 : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The time-space part of the metric.
template <typename DataType, size_t Dim, typename Frame>
struct Metrick0 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/// The time-space part of the generalized harmonic variable \f$Pi\f$
template <typename DataType, size_t Dim, typename Frame>
struct Pik0 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/// The space-space part of the metric.
template <typename DataType, size_t Dim, typename Frame>
struct Metrickj : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/// The space-space part of the generalized harmonic variable \f$Pi\f$
template <typename DataType, size_t Dim, typename Frame>
struct Pikj : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/// \f$\Phi_{k00}, where \f$\Phi\f$ is the generalized harmonic variable.
template <typename DataType, size_t Dim, typename Frame>
struct Phik00 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/// \f$\Phi_{ki0}, where \f$\Phi\f$ is the generalized harmonic variable.
template <typename DataType, size_t Dim, typename Frame>
struct Phiki0 : db::SimpleTag {
  using type = tnsr::ij<DataType, Dim, Frame>;
};

/// \f$\Phi_{kij}, where \f$\Phi\f$ is the generalized harmonic variable.
template <typename DataType, size_t Dim, typename Frame>
struct Phikij : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

}  // namespace Tags

using gh_spacetime_vars_list =
    tmpl::list<::gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>,
               ::gh::Tags::Pi<DataVector, 3, Frame::Inertial>,
               ::gh::Tags::Phi<DataVector, 3, Frame::Inertial>>;

template <typename Frame>
using gh_spatial_vars_list = tmpl::list<
    Tags::Metric00<DataVector>, Tags::Metrick0<DataVector, 3, Frame>,
    Tags::Metrickj<DataVector, 3, Frame>, Tags::Pi00<DataVector>,
    Tags::Pik0<DataVector, 3, Frame>, Tags::Pikj<DataVector, 3, Frame>,
    Tags::Phik00<DataVector, 3, Frame>, Tags::Phiki0<DataVector, 3, Frame>,
    Tags::Phikij<DataVector, 3, Frame>>;

/*!
 * \brief Copies spacetime variables into their spatial pieces.
 *
 * For example, if one of the spacetime variables is the metric
 * $g_{ab}$, then the corresponding spatial variables are a spatial
 * scalar $g_{00}$, a spatial vector $g_{i0}$, and a spatial symmetric
 * 2-tensor $g_{ij}$.
 *
 * The arguments must already be allocated to their correct sizes; no
 * memory allocation is done.
 *
 * \param spatial_vars Points to a Variables containing spatial pieces.
 * \param spacetime_vars A Variables containing the spacetime variables.
 */
void break_spacetime_vars_into_spatial_pieces(
    gsl::not_null<Variables<gh_spatial_vars_list<Frame::Inertial>>*>
        spatial_vars,
    const Variables<gh_spacetime_vars_list>& spacetime_vars);

/*!
 * \brief Copies spatial pieces into the corresponding spacetime variables.
 *
 * This is the inverse of break_spacetime_vars_into_spatial_pieces.
 *
 * \param spatial_vars A Variables containing spatial pieces.
 * \param spacetime_vars A Variables containing the spacetime variables.
 */
void assemble_spacetime_vars_from_spatial_pieces(
    gsl::not_null<Variables<gh_spacetime_vars_list>*> spacetime_vars,
    const Variables<gh_spatial_vars_list<Frame::Inertial>>& spatial_vars);

/*!
 * \brief Transforms spatial tensors into a different frame, ignoring hessians.
 *
 * This is done for filtering, where having the correct (i.e. with hessians)
 * transformation doesn't matter; all that matters is that the tensor
 * indices correspond to the coordinates (or in other words, no dual frame).
 *
 * Assumes that all the variables have lower indices.
 *
 * Takes special care to re-use memory. The Variables arguments must
 * already be allocated to their correct sizes; no memory allocation
 * is done.
 *
 * \tparam SrcFrame Source frame.
 * \tparam DestFrame Destination frame.
 * \param dest A Variables for the destination spatial variables.
 * \param src A Variables containing the source spatial variables.
 * \param jac The jacobian dx^src/dx^dest
 */
template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    gsl::not_null<Variables<gh_spatial_vars_list<DestFrame>>*> dest,
    const Variables<gh_spatial_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac);

}  // namespace filter_detail

/*!
 * \brief Applies TensorYlm filter in place to Generalized Harmonic variables.
 *
 * When radial_extents is 1, gh_vars and temp_storage are assumed to
 * be defined on a spherical slice, with number of grid points
 * corresponding to a spherical-harmonic grid of ell_max, and the
 * filter happens only on that slice.
 *
 * When radial_extents is > 1, gh_vars and temp_storage are assumed to
 * be defined on a spherical shell of topology I1 x S2. The filter
 * happens in the entire volume, internally iterating over each
 * spherical slice at a time.
 *
 * For performance reasons, apply_tensor_ylm_filter does not allocate
 * or deallocate memory, but it does take a temp_storage buffer.  The
 * size of temp_storage should at least
 * radial_extents*spectral_size*num_components, where num_components
 * is the total number of independent components in the GH variable
 * list (i.e. 30), and spectral_size is the size of the S2 Spherepack
 * spectral coefficient array for ell_max, as obtained from the member
 * function ylm::Spherepack::spectral_size().  Note that for S2 on
 * Spherepack, the number of collocation points is different than the
 * number of spectral coefficients, and both are different than the
 * size of the Spherepack storage array.
 *
 * \param gh_vars Generalized Harmonic variables at collocation points.
 * \param temp_storage Temporary storage for generalized harmonic variables,
 *   allocated outside apply_tensor_ylm_filter. See above for size requirements.
 * \param jac_inertial_to_grid Jacobian taking V_x from inertial to grid.
 * \param jac_grid_to_inertial Jacobian taking V_x from grid to inertial.
 * \param filter_matrix_scalar The scalar filter matrix computed by fill_filter.
 * \param filter_matrix_i The Rank-1 matrix computed by fill_filter.
 * \param filter_matrix_ii The Rank-2 symmetric matrix computed by fill_filter.
 * \param filter_matrix_ij The Rank-2 matrix computed by fill_filter.
 * \param filter_matrix_kii The Rank-3 matrix computed by fill_filter.
 * \param ell_max The maximum ylm ell.
 * \param radial_extents The number of radial grid points, can be 1 for slices.
 */
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<filter_detail::gh_spacetime_vars_list>*> gh_vars,
    gsl::not_null<Variables<filter_detail::gh_spacetime_vars_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const SimpleSparseMatrix& filter_matrix_scalar,
    const SimpleSparseMatrix& filter_matrix_i,
    const SimpleSparseMatrix& filter_matrix_ii,
    const SimpleSparseMatrix& filter_matrix_ij,
    const SimpleSparseMatrix& filter_matrix_kii, size_t ell_max,
    size_t radial_extents);

class TensorYlmFilter {
 public:
  struct NumModesToKill {
    using type = size_t;
    static constexpr Options::String help =
        "Number of highest ell modes to zero in the TensorYlm filter.";
  };

  struct HalfPower {
    using type = size_t;
    static constexpr Options::String help =
        "Half power for smooth TensorYlm filtering.";
    static type lower_bound() { return 1; }
  };

  struct Enable {
    using type = bool;
    static constexpr Options::String help = "Enable the TensorYlm filter.";
  };

  struct FilterEveryNSlabs {
    using type = size_t;
    static constexpr Options::String help = {
        "Apply the TensorYlm filter every N slabs during evolution."};
    static type lower_bound() { return 1; }
    static type suggested_value() { return 1; }
  };

  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to apply TensorYlm filtering to. "
        "Use 'All' to filter all blocks."};
  };

  using options = tmpl::list<NumModesToKill, HalfPower, Enable,
                             FilterEveryNSlabs, BlocksToFilter>;
  static constexpr Options::String help = {"A TensorYlm filter."};
  static std::string name() { return "TensorYlmFilter"; }

  TensorYlmFilter() = default;
  TensorYlmFilter(
      size_t num_modes_to_kill, size_t half_power, bool enable,
      size_t filter_every_n_slabs,
      const std::optional<std::vector<std::string>>& blocks_to_filter,
      const Options::Context& context = {});

  size_t num_modes_to_kill() const { return num_modes_to_kill_; }
  std::optional<size_t> half_power() const { return half_power_; }
  bool enable() const { return enable_; }
  size_t filter_every_n_slabs() const { return filter_every_n_slabs_; }
  const std::optional<std::unordered_set<std::string>>& blocks_to_filter()
      const {
    return blocks_to_filter_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const TensorYlmFilter& lhs,
                         const TensorYlmFilter& rhs);

  size_t num_modes_to_kill_{0};
  std::optional<size_t> half_power_{32};
  bool enable_{false};
  size_t filter_every_n_slabs_{1};
  std::optional<std::unordered_set<std::string>> blocks_to_filter_{};
};

bool operator==(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs);
bool operator!=(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs);
}  // namespace ylm::TensorYlm

namespace gh::Actions {

namespace detail {

struct TensorYlmFilterMatrixCacheKey {
  size_t ell_max;
  size_t num_modes_to_kill;
  std::optional<size_t> half_power;

  bool operator==(const TensorYlmFilterMatrixCacheKey& other) const {
    return ell_max == other.ell_max and
           num_modes_to_kill == other.num_modes_to_kill and
           half_power == other.half_power;
  }
};

struct TensorYlmFilterMatrixCacheKeyHash {
  size_t operator()(const TensorYlmFilterMatrixCacheKey& key) const {
    size_t hash = 0;
    hash ^= std::hash<size_t>{}(key.ell_max) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    hash ^= std::hash<size_t>{}(key.num_modes_to_kill) + 0x9e3779b9 +
            (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>{}(key.half_power.has_value()) + 0x9e3779b9 +
            (hash << 6) + (hash >> 2);
    if (key.half_power.has_value()) {
      hash ^= std::hash<size_t>{}(key.half_power.value()) + 0x9e3779b9 +
              (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

struct TensorYlmFilterMatrices {
  SimpleSparseMatrix scalar{};
  SimpleSparseMatrix i{};
  SimpleSparseMatrix ii{};
  SimpleSparseMatrix ij{};
  SimpleSparseMatrix kii{};
};

inline const TensorYlmFilterMatrices& cached_filter_matrices(
    const size_t ell_max, const size_t num_modes_to_kill,
    const std::optional<size_t>& half_power) {
  static std::unordered_map<TensorYlmFilterMatrixCacheKey,
                            TensorYlmFilterMatrices,
                            TensorYlmFilterMatrixCacheKeyHash>
      cache{};
  static std::mutex cache_mutex{};

  const TensorYlmFilterMatrixCacheKey key{ell_max, num_modes_to_kill,
                                          half_power};
  {
    const std::lock_guard<std::mutex> lock(cache_mutex);
    if (const auto iter = cache.find(key); iter != cache.end()) {
      return iter->second;
    }
  }

  TensorYlmFilterMatrices matrices{};
  ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
      make_not_null(&matrices.scalar), ell_max, num_modes_to_kill, half_power,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_filter<tnsr::i<DataVector, 3>::structure>(
      make_not_null(&matrices.i), ell_max, num_modes_to_kill, half_power,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_filter<tnsr::ii<DataVector, 3>::structure>(
      make_not_null(&matrices.ii), ell_max, num_modes_to_kill, half_power,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_filter<tnsr::ij<DataVector, 3>::structure>(
      make_not_null(&matrices.ij), ell_max, num_modes_to_kill, half_power,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_filter<tnsr::ijj<DataVector, 3>::structure>(
      make_not_null(&matrices.kii), ell_max, num_modes_to_kill, half_power,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);

  const std::lock_guard<std::mutex> lock(cache_mutex);
  const auto [iter, inserted] = cache.emplace(key, std::move(matrices));
  return iter->second;
}

}  // namespace detail

struct ApplyTensorYlmFilter {
  using const_global_cache_tags =
      tmpl::list<Filters::Tags::Filter<ylm::TensorYlm::TensorYlmFilter>,
                 domain::Tags::Domain<3>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/);
};

template <typename DbTagsList, typename... InboxTags, typename Metavariables,
          typename ArrayIndex, typename ActionList, typename ParallelComponent>
Parallel::iterable_action_return_t ApplyTensorYlmFilter::apply(
    db::DataBox<DbTagsList>& box,
    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
    const Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
    const ParallelComponent* const /*meta*/) {
  const auto& filter =
      Parallel::get<Filters::Tags::Filter<ylm::TensorYlm::TensorYlmFilter>>(
          cache);
  if (not filter.enable()) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
  const auto slab_number = db::get<::Tags::TimeStepId>(box).slab_number();
  if (slab_number >= 0 and
      static_cast<size_t>(slab_number) % filter.filter_every_n_slabs() != 0) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

  const size_t block_id =
      db::get<domain::Tags::Element<3>>(box).id().block_id();
  const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
  const auto& block_groups = domain.block_groups();
  const std::string& block_name = domain.blocks()[block_id].name();
  if (filter.blocks_to_filter().has_value()) {
    const bool filter_this_block = alg::any_of(
        filter.blocks_to_filter().value(),
        [&block_name, &block_groups](const std::string& block_to_filter) {
          return domain::block_is_in_group(block_name, block_to_filter,
                                           block_groups);
        });
    if (not filter_this_block) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
  }

  const Mesh<3>& mesh = db::get<domain::Tags::Mesh<3>>(box);
  if (mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(2) != Spectral::Basis::SphericalHarmonic) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

  const size_t ell_max = mesh.extents(1) - 1;
  const size_t radial_extents = mesh.extents(0);
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto& filter_matrices = detail::cached_filter_matrices(
      ell_max, filter.num_modes_to_kill(), filter.half_power());

  Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list> temp_storage(
      radial_extents * ylm::Spherepack::spectral_size(ell_max, ell_max), 0.0);
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
      jac_inertial_to_grid(number_of_grid_points, 0.0);
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      jac_grid_to_inertial(number_of_grid_points, 0.0);
  {
    const auto& inv_jac_logical_to_inertial =
        db::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(box);
    const auto& inv_jac_logical_to_grid = db::get<
        domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Grid>>(
        box);
    Jacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        jac_logical_to_inertial(number_of_grid_points, 0.0);
    Scalar<DataVector> det(number_of_grid_points, 0.0);
    determinant_and_inverse(make_not_null(&det),
                            make_not_null(&jac_logical_to_inertial),
                            inv_jac_logical_to_inertial);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        jac_inertial_to_grid.get(i, j) = jac_logical_to_inertial.get(i, 0) *
                                             inv_jac_logical_to_grid.get(0, j) +
                                         jac_logical_to_inertial.get(i, 1) *
                                             inv_jac_logical_to_grid.get(1, j) +
                                         jac_logical_to_inertial.get(i, 2) *
                                             inv_jac_logical_to_grid.get(2, j);
      }
    }
    determinant_and_inverse(make_not_null(&det),
                            make_not_null(&jac_grid_to_inertial),
                            jac_inertial_to_grid);
  }
  db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
             gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>(
      [&temp_storage, &jac_inertial_to_grid, &jac_grid_to_inertial,
       &filter_matrices, ell_max, radial_extents](
          const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
          const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
          const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi) {
        Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list> vars(
            spacetime_metric->begin()->size());
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars) = *spacetime_metric;
        get<gh::Tags::Pi<DataVector, 3>>(vars) = *pi;
        get<gh::Tags::Phi<DataVector, 3>>(vars) = *phi;
        ylm::TensorYlm::apply_tensor_ylm_filter(
            make_not_null(&vars), make_not_null(&temp_storage),
            jac_inertial_to_grid, jac_grid_to_inertial, filter_matrices.scalar,
            filter_matrices.i, filter_matrices.ii, filter_matrices.ij,
            filter_matrices.kii, ell_max, radial_extents);
        *spacetime_metric = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(vars);
        *pi = get<gh::Tags::Pi<DataVector, 3>>(vars);
        *phi = get<gh::Tags::Phi<DataVector, 3>>(vars);
      },
      make_not_null(&box));

  return {Parallel::AlgorithmExecution::Continue, std::nullopt};
}

}  // namespace gh::Actions
