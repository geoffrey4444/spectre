// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
class NodeLock;
}  // namespace Parallel
namespace observers::Tags {
struct H5FileLock;
}  // namespace observers::Tags
/// \endcond

namespace intrp::callbacks {
namespace option_tags {

struct InitialAdmEnergy {
  using type = Options::Auto<double>;
  static constexpr Options::String help = {
      "Initial ADM energy metadata to write into finite-radius RWZ files. "
      "If set to 'Auto', the native RWZ output stores NaN for this quantity."};
};

}  // namespace option_tags

namespace cache_tags {

struct InitialAdmEnergy : db::SimpleTag {
  using type = std::optional<double>;
  using option_tags = tmpl::list<option_tags::InitialAdmEnergy>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

}  // namespace cache_tags

namespace detail {

inline std::vector<std::string> rwz_legend(const size_t l_max) {
  std::vector<std::string> legend{};
  legend.reserve(1 + 2 * square(l_max + 1));
  legend.emplace_back("Time");
  for (int l = 0; l <= static_cast<int>(l_max); ++l) {
    for (int m = -l; m <= l; ++m) {
      legend.push_back(MakeString{} << "Re(" << l << "," << m << ")");
      legend.push_back(MakeString{} << "Im(" << l << "," << m << ")");
    }
  }
  return legend;
}

inline std::vector<double> rwz_row(const ComplexModalVector& modal_data,
                                   const size_t l_max, const double time) {
  std::vector<double> row{};
  row.reserve(1 + 2 * square(l_max + 1));
  row.push_back(time);
  for (size_t i = 0; i < square(l_max + 1); ++i) {
    row.push_back(real(modal_data[i]));
    row.push_back(imag(modal_data[i]));
  }
  return row;
}

inline std::string radius_label(const double radius) {
  std::ostringstream stream{};
  stream << std::setprecision(14) << radius;
  auto label = stream.str();
  for (auto& c : label) {
    if (c == '.') {
      c = '_';
    }
  }
  return label;
}

inline void write_rwz_quantity(
    const gsl::not_null<h5::H5File<h5::AccessType::ReadWrite>*> output_file,
    const std::string& subfile_name, const ComplexModalVector& modal_data,
    const size_t l_max, const double time) {
  auto& dataset =
      output_file->try_insert<h5::Dat>(subfile_name, rwz_legend(l_max), 0);
  dataset.append(rwz_row(modal_data, l_max, time));
  output_file->close_current_object();
}

inline void write_scalar_quantity(
    const gsl::not_null<h5::H5File<h5::AccessType::ReadWrite>*> output_file,
    const std::string& subfile_name, const std::string& legend_name,
    const double value, const double time) {
  auto& dataset = output_file->try_insert<h5::Dat>(
      subfile_name, std::vector<std::string>{"Time", legend_name}, 0);
  dataset.append(std::vector<double>{time, value});
  output_file->close_current_object();
}

}  // namespace detail

template <typename InterpolationTargetTag>
struct ObserveReggeWheelerZerilli
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  using const_global_cache_tags = tmpl::list<observers::Tags::SurfaceFileName,
                                             cache_tags::InitialAdmEnergy>;

  using gh_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;
  using finite_radius_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                 gr::Tags::SpatialRicci<DataVector, 3, Frame::Inertial>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataVector, 3,
                                                            Frame::Inertial>,
                               tmpl::size_t<3>, Frame::Inertial>>;

  static_assert(
      std::is_same_v<
          typename InterpolationTargetTag::vars_to_interpolate_to_target,
          finite_radius_source_vars>,
      "ObserveReggeWheelerZerilli requires the finite-radius GH extraction "
      "payload to be interpolated to the target.");

  static_assert(
      std::is_same_v<typename InterpolationTargetTag::compute_target_points,
                     intrp::TargetPoints::Sphere<InterpolationTargetTag,
                                                 ::Frame::Inertial>>,
      "ObserveReggeWheelerZerilli requires an inertial Sphere target.");

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    auto* hdf5_lock = Parallel::local_synchronous_action<
        observers::Actions::GetLockPointer<observers::Tags::H5FileLock>>(
        Parallel::get_parallel_component<
            observers::ObserverWriter<Metavariables>>(cache));

    const auto& sphere =
        Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
    if (sphere.angular_ordering != ylm::AngularOrdering::Strahlkorper) {
      ERROR(
          "ObserveReggeWheelerZerilli currently requires "
          "AngularOrdering: Strahlkorper, not "
          << sphere.angular_ordering);
    }

    const double time =
        intrp::InterpolationTarget_detail::get_temporal_id_value(temporal_id);
    const size_t l_max = sphere.l_max;
    const size_t num_points_single_sphere = (l_max + 1) * (2 * l_max + 1);
    const auto& all_spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(box);
    const auto& all_pi = get<gh::Tags::Pi<DataVector, 3>>(box);
    const auto& all_phi = get<gh::Tags::Phi<DataVector, 3>>(box);
    const auto& all_spatial_ricci =
        get<gr::Tags::SpatialRicci<DataVector, 3, Frame::Inertial>>(box);
    const auto& all_extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>>(box);
    const auto& all_cov_deriv_extrinsic_curvature = get<::Tags::deriv<
        gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(box);
    const auto& all_coords = get<Tags::AllCoords<::Frame::Inertial>>(box);

    const std::string filename =
        Parallel::get<observers::Tags::SurfaceFileName>(cache) + ".h5";
    const double initial_adm_energy =
        Parallel::get<cache_tags::InitialAdmEnergy>(cache).value_or(
            std::numeric_limits<double>::quiet_NaN());
    const std::lock_guard lock(*hdf5_lock);
    h5::H5File<h5::AccessType::ReadWrite> output_file(filename, true);

    size_t offset = 0;
    for (const double radius : sphere.radii) {
      const tnsr::aa<DataVector, 3, ::Frame::Inertial> spacetime_metric;
      const tnsr::aa<DataVector, 3, ::Frame::Inertial> pi;
      const tnsr::iaa<DataVector, 3, ::Frame::Inertial> phi;
      const tnsr::ii<DataVector, 3, ::Frame::Inertial> spatial_ricci;
      const tnsr::ii<DataVector, 3, ::Frame::Inertial> extrinsic_curvature;
      const tnsr::ijj<DataVector, 3, ::Frame::Inertial>
          cov_deriv_extrinsic_curvature;
      const tnsr::I<DataVector, 3, ::Frame::Inertial> coords;

      for (size_t a = 0; a < 4; ++a) {
        for (size_t b = 0; b < 4; ++b) {
          make_const_view(make_not_null(&spacetime_metric.get(a, b)),
                          all_spacetime_metric.get(a, b), offset,
                          num_points_single_sphere);
          make_const_view(make_not_null(&pi.get(a, b)), all_pi.get(a, b),
                          offset, num_points_single_sphere);
          for (size_t i = 0; i < 3; ++i) {
            make_const_view(make_not_null(&phi.get(i, a, b)),
                            all_phi.get(i, a, b), offset,
                            num_points_single_sphere);
          }
        }
      }
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          make_const_view(make_not_null(&spatial_ricci.get(i, j)),
                          all_spatial_ricci.get(i, j), offset,
                          num_points_single_sphere);
          make_const_view(make_not_null(&extrinsic_curvature.get(i, j)),
                          all_extrinsic_curvature.get(i, j), offset,
                          num_points_single_sphere);
          for (size_t k = 0; k < 3; ++k) {
            make_const_view(
                make_not_null(&cov_deriv_extrinsic_curvature.get(k, i, j)),
                all_cov_deriv_extrinsic_curvature.get(k, i, j), offset,
                num_points_single_sphere);
          }
        }
      }
      for (size_t i = 0; i < 3; ++i) {
        make_const_view(make_not_null(&coords.get(i)), all_coords.get(i),
                        offset, num_points_single_sphere);
      }

      const ylm::Strahlkorper<::Frame::Inertial> strahlkorper{
          l_max, l_max, DataVector{num_points_single_sphere, radius},
          sphere.center};
      const auto rwz =
          gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
              spacetime_metric, pi, phi, coords, strahlkorper.ylm_spherepack(),
              sphere.center, radius);
      const auto r_times_psi_4 = gr::surfaces::psi_4_modes_from_tensors(
          spacetime_metric, spatial_ricci, extrinsic_curvature,
          cov_deriv_extrinsic_curvature, coords, strahlkorper.ylm_spherepack(),
          sphere.center, radius);
      const auto metadata =
          gr::surfaces::extraction_sphere_metadata_from_gh_vars(
              spacetime_metric, strahlkorper);

      const std::string base_subfile =
          "/" + pretty_type::name<InterpolationTargetTag>() + "/Radius" +
          detail::radius_label(radius);
      detail::write_scalar_quantity(make_not_null(&output_file),
                                    base_subfile + "/CoordRadius",
                                    "CoordRadius", radius, time);
      detail::write_scalar_quantity(
          make_not_null(&output_file), base_subfile + "/InitialAdmEnergy",
          "InitialAdmEnergy", initial_adm_energy, time);
      detail::write_scalar_quantity(
          make_not_null(&output_file), base_subfile + "/AverageLapse",
          "AverageLapse", metadata.average_lapse, time);
      detail::write_scalar_quantity(make_not_null(&output_file),
                                    base_subfile + "/ArealRadius",
                                    "ArealRadius", metadata.areal_radius, time);
      detail::write_rwz_quantity(make_not_null(&output_file),
                                 base_subfile + "/Strain", rwz.r_times_strain,
                                 l_max, time);
      detail::write_rwz_quantity(make_not_null(&output_file),
                                 base_subfile + "/PhiPlus", rwz.phi_plus, l_max,
                                 time);
      detail::write_rwz_quantity(make_not_null(&output_file),
                                 base_subfile + "/PhiMinus", rwz.phi_minus,
                                 l_max, time);
      detail::write_rwz_quantity(make_not_null(&output_file),
                                 base_subfile + "/Psi4", r_times_psi_4, l_max,
                                 time);

      offset += num_points_single_sphere;
    }
  }
};

}  // namespace intrp::callbacks
