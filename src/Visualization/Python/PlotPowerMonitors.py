# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from itertools import cycle
from typing import Iterable, Optional, Sequence, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr
from spectre.Domain import Domain, deserialize_domain
from spectre.IO.H5 import open_volfiles, open_volfiles_command, parse_point
from spectre.IO.H5.IterElements import iter_elements, stripped_element_name
from spectre.NumericalAlgorithms.LinearOperators import (
    power_monitors,
    shell_power_monitor_buffer,
    shell_power_monitors,
)
from spectre.Spectral import Basis
from spectre.support.CliExceptions import RequiredChoiceError
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)

logger = logging.getLogger(__name__)


def _has_shell_topology(mesh) -> bool:
    basis = mesh.basis()
    return (
        mesh.dim == 3
        and sum(b == Basis.SphericalHarmonic for b in basis) == 2
        and sum(b != Basis.SphericalHarmonic for b in basis) == 1
    )


def is_shell_mesh(mesh) -> bool:
    basis = mesh.basis()
    return (
        _has_shell_topology(mesh)
        and basis[0] != Basis.SphericalHarmonic
        and basis[1] == Basis.SphericalHarmonic
        and basis[2] == Basis.SphericalHarmonic
    )


def _check_supported_shell_layout(mesh):
    if _has_shell_topology(mesh) and not is_shell_mesh(mesh):
        raise ValueError(
            "Shell power monitors currently support only meshes with the "
            "radial dimension first and the two spherical-harmonic "
            "dimensions last."
        )


def _check_compatible_monitor_labels(monitor_labels, labels):
    if monitor_labels is None or labels == monitor_labels:
        return
    raise ValueError(
        "Plotting power monitors for domains that mix spherical-shell "
        "elements with non-shell elements is not yet supported."
    )


def _combine_shell_buffers(buffers):
    combined = None
    for buffer in buffers:
        if combined is None:
            combined = {
                key: np.array(value, copy=True) for key, value in buffer.items()
            }
            continue
        combined["radial_sums"] += np.array(buffer["radial_sums"])
        combined["angular_sums"] += np.array(buffer["angular_sums"])
        combined["radial_counts"] += np.array(buffer["radial_counts"])
        combined["angular_counts"] += np.array(buffer["angular_counts"])
    if combined is None:
        return None
    return combined


def _finalize_shell_buffer(buffer):
    radial = np.zeros(len(buffer["radial_sums"]))
    angular = np.zeros(len(buffer["angular_sums"]))
    radial_mask = np.array(buffer["radial_counts"]) > 0
    angular_mask = np.array(buffer["angular_counts"]) > 0
    radial[radial_mask] = np.sqrt(
        np.array(buffer["radial_sums"])[radial_mask]
        / np.array(buffer["radial_counts"])[radial_mask]
    )
    angular[angular_mask] = np.sqrt(
        np.array(buffer["angular_sums"])[angular_mask]
        / np.array(buffer["angular_counts"])[angular_mask]
    )
    return [radial, angular]


def _tensor_data_to_spatial_tensor(
    spatial_suffixes, spatial_data, rank, symmetric_last_two=False
):
    num_points = spatial_data.shape[1]
    if rank == 1:
        return tnsr.i[DataVector, 3](spatial_data)
    if rank == 2:
        tensor_type = tnsr.ii if symmetric_last_two else tnsr.ij
        tensor = tensor_type[DataVector, 3](num_points)
    elif rank == 3 and symmetric_last_two:
        tensor = tnsr.ijj[DataVector, 3](num_points)
    else:
        raise ValueError(
            f"Unsupported spatial tensor rank {rank} for shell power monitor."
        )
    index_map = {"x": 0, "y": 1, "z": 2}
    for suffix, component in zip(spatial_suffixes, spatial_data):
        indices = tuple(index_map[index] for index in suffix)
        storage_index = tensor.get_storage_index(*indices)
        tensor[storage_index] = DataVector(component)
    return tensor


def _expand_symmetric_last_two_components(spatial_suffixes, spatial_data):
    expanded_suffixes = []
    expanded_data = []
    for suffix, component in zip(spatial_suffixes, spatial_data):
        expanded_suffixes.append(suffix)
        expanded_data.append(component)
        if suffix[-2] != suffix[-1]:
            swapped_suffix = suffix[:-2] + suffix[-1] + suffix[-2]
            expanded_suffixes.append(swapped_suffix)
            expanded_data.append(component)
    return expanded_suffixes, np.array(expanded_data)


def _shell_buffers_for_tensor_variable(component_names, tensor_data, mesh):
    suffixes = [
        (
            component_name.split("_", maxsplit=1)[1]
            if "_" in component_name
            else ""
        )
        for component_name in component_names
    ]
    if suffixes == [""]:
        return [shell_power_monitor_buffer(DataVector(tensor_data[0]), mesh)]

    grouped_components = {}
    for suffix, component in zip(suffixes, tensor_data):
        spatial_suffix = suffix.replace("t", "")
        key = (suffix.count("t"), len(spatial_suffix))
        grouped_components.setdefault(key, []).append(
            (spatial_suffix, component)
        )

    buffers = []
    for (_, spatial_rank), grouped_data in grouped_components.items():
        spatial_suffixes = [suffix for suffix, _ in grouped_data]
        spatial_data = np.array([component for _, component in grouped_data])
        if spatial_rank == 0:
            if len(spatial_data) != 1:
                raise ValueError(
                    "Expected exactly one scalar component in shell tensor "
                    f"decomposition, but got {len(spatial_data)}."
                )
            buffers.append(
                shell_power_monitor_buffer(
                    Scalar[DataVector](spatial_data[0]), mesh
                )
            )
            continue
        if spatial_rank == 1:
            buffers.append(
                shell_power_monitor_buffer(
                    _tensor_data_to_spatial_tensor(
                        spatial_suffixes, spatial_data, rank=1
                    ),
                    mesh,
                )
            )
            continue
        if spatial_rank == 2:
            symmetric_last_two = len(spatial_data) == 6 and all(
                len(suffix) == 2 for suffix in spatial_suffixes
            )
            buffers.append(
                shell_power_monitor_buffer(
                    _tensor_data_to_spatial_tensor(
                        spatial_suffixes,
                        spatial_data,
                        rank=2,
                        symmetric_last_two=symmetric_last_two,
                    ),
                    mesh,
                )
            )
            continue
        if spatial_rank == 3:
            symmetric_last_two = len(spatial_data) == 18 and all(
                len(suffix) == 3 for suffix in spatial_suffixes
            )
            buffers.append(
                shell_power_monitor_buffer(
                    _tensor_data_to_spatial_tensor(
                        spatial_suffixes,
                        spatial_data,
                        rank=3,
                        symmetric_last_two=symmetric_last_two,
                    ),
                    mesh,
                )
            )
            continue
        raise ValueError(
            "Unsupported shell tensor decomposition for components "
            f"{component_names}."
        )
    return buffers


def _component_monitors(tensor_data, mesh, skip_filtered_modes):
    _check_supported_shell_layout(mesh)
    if is_shell_mesh(mesh):
        all_modes = None
        for component in tensor_data:
            component_monitors = shell_power_monitors(
                DataVector(component), mesh
            )
            component_modes = [
                np.array(component_monitors["radial"])[
                    : -skip_filtered_modes or None
                ],
                np.array(component_monitors["angular"])[
                    : -skip_filtered_modes or None
                ],
            ]
            if all_modes is None:
                all_modes = [
                    np.zeros_like(component_modes[0]),
                    np.zeros_like(component_modes[1]),
                ]
            for i, component_mode in enumerate(component_modes):
                all_modes[i] += component_mode**2
        return ["radial", "angular-l"], [np.sqrt(modes) for modes in all_modes]

    all_modes = [
        np.zeros(mesh.extents(d) - skip_filtered_modes) for d in range(mesh.dim)
    ]
    for component in tensor_data:
        modes = power_monitors(DataVector(component), mesh)
        for d, modes_dim in enumerate(modes):
            num_modes = len(modes_dim) - skip_filtered_modes
            all_modes[d] += np.array(modes_dim)[:num_modes] ** 2
    return None, [np.sqrt(modes_dim) for modes_dim in all_modes]


def _shell_component_average_radial_monitor(
    tensor_data, mesh, skip_filtered_modes
):
    radial_modes = None
    for component in tensor_data:
        component_monitors = shell_power_monitors(DataVector(component), mesh)
        component_radial = np.array(component_monitors["radial"])[
            : -skip_filtered_modes or None
        ]
        if radial_modes is None:
            radial_modes = np.zeros_like(component_radial)
        radial_modes += component_radial**2
    return np.sqrt(radial_modes / len(tensor_data))


def _tensor_variable_monitors(
    component_names, tensor_data, mesh, skip_filtered_modes
):
    _check_supported_shell_layout(mesh)
    if is_shell_mesh(mesh):
        shell_buffer = _combine_shell_buffers(
            _shell_buffers_for_tensor_variable(
                component_names, tensor_data, mesh
            )
        )
        all_modes = _finalize_shell_buffer(shell_buffer)
        if skip_filtered_modes:
            all_modes = [
                modes[:-skip_filtered_modes] if skip_filtered_modes else modes
                for modes in all_modes
            ]
        all_modes[0] = _shell_component_average_radial_monitor(
            tensor_data, mesh, skip_filtered_modes
        )
        return ["radial", "angular-l"], all_modes

    all_modes = [
        np.zeros(mesh.extents(d) - skip_filtered_modes) for d in range(mesh.dim)
    ]
    for component in tensor_data:
        modes = power_monitors(DataVector(component), mesh)
        for d, modes_dim in enumerate(modes):
            num_modes = len(modes_dim) - skip_filtered_modes
            all_modes[d] += np.array(modes_dim)[:num_modes] ** 2
    return None, [
        np.sqrt(modes_dim / len(tensor_data)) for modes_dim in all_modes
    ]


def _select_tensor_components(all_components, tensor_variable):
    selected = [
        component
        for component in all_components
        if component == tensor_variable
        or component.startswith(tensor_variable + "_")
    ]
    if not selected:
        raise RequiredChoiceError(
            f"'{tensor_variable}' matches no tensor variable.",
            choices=all_components,
        )
    return selected


def find_block_or_group(
    block_id: int,
    block_or_group_names: Sequence[str],
    domain: Union[Domain[1], Domain[2], Domain[3]],
) -> Optional[int]:
    """Find entry in 'block_or_group_names' that corresponds to the 'block_id'"""
    block_name = domain.blocks[block_id].name
    for i, name in enumerate(block_or_group_names):
        if name == block_name:
            return i
        if (
            name in domain.block_groups
            and block_name in domain.block_groups[name]
        ):
            return i
    return None


def plot_power_monitors(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_id: Optional[int],
    tensor_components: Sequence[str],
    tensor_variable: Optional[str],
    block_or_group_names: Sequence[str],
    domain: Union[Domain[1], Domain[2], Domain[3]],
    dimension_labels: Sequence[str] = [r"$\xi$", r"$\eta$", r"$\zeta$"],
    element_patterns: Optional[Sequence[str]] = None,
    skip_filtered_modes: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
):
    plot_over_time = obs_id is None
    num_cols = len(block_or_group_names)
    monitor_labels = None
    all_mode_time_series = None
    direct_mode_data = None
    num_elements = None
    max_error = None
    separate_rows = plot_over_time

    shown_dtype_warning_once = False
    for element, tensor_data in iter_elements(
        volfiles, obs_id, tensor_components, element_patterns=element_patterns
    ):
        # Skip FD elements because we can't compute power monitors for them
        if any(
            basis == Basis.FiniteDifference for basis in element.mesh.basis()
        ):
            continue

        # Find the subplot for this element's block, or skip the element if its
        # block wasn't selected
        subplot_index = find_block_or_group(
            element.id.block_id, block_or_group_names, domain
        )
        if subplot_index is None:
            continue

        if tensor_data.dtype != np.float64:
            if not shown_dtype_warning_once:
                logger.warning(
                    "Tensor data is not double precision. Power monitors"
                    " will be inaccurate below the precision of the data."
                )
                shown_dtype_warning_once = True
            tensor_data = tensor_data.astype(np.float64)

        _check_supported_shell_layout(element.mesh)

        try:
            labels, all_modes = (
                _tensor_variable_monitors(
                    tensor_components,
                    tensor_data,
                    element.mesh,
                    skip_filtered_modes,
                )
                if tensor_variable is not None
                else _component_monitors(
                    tensor_data, element.mesh, skip_filtered_modes
                )
            )
        except ValueError as error:
            logger.warning(
                (
                    "Skipping element '%s' because power monitor computation "
                    "failed: %s"
                ),
                element.id,
                error,
            )
            continue
        labels = labels or list(dimension_labels[: len(all_modes)])

        if monitor_labels is None:
            monitor_labels = labels
            separate_rows = plot_over_time or labels == ["radial", "angular-l"]
            if plot_over_time:
                all_mode_time_series = {
                    subplot_index: dict() for subplot_index in range(num_cols)
                }
            else:
                direct_mode_data = {
                    subplot_index: [] for subplot_index in range(num_cols)
                }
                num_elements = np.zeros(num_cols, dtype=int)
                max_error = np.zeros((num_cols, len(monitor_labels)))
        else:
            _check_compatible_monitor_labels(monitor_labels, labels)

        if plot_over_time:
            all_mode_time_series[subplot_index].setdefault(
                element.id, []
            ).append((element.time, all_modes))
        else:
            direct_mode_data[subplot_index].append(all_modes)
            num_elements[subplot_index] += 1

    if monitor_labels is None:
        raise ValueError("No supported elements were found for power monitors.")

    num_rows = len(monitor_labels) if separate_rows else 1
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=figsize or (num_cols * 4, num_rows * 4),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    prop_cycle = {
        key: cycle(values)
        for key, values in plt.rcParams["axes.prop_cycle"].by_key().items()
    }
    props_dim = {
        d: {key: next(values) for key, values in prop_cycle.items()}
        for d in range(len(monitor_labels))
    }

    if not plot_over_time:
        for subplot_index in range(num_cols):
            for all_modes in direct_mode_data[subplot_index]:
                for d, modes_dim in enumerate(all_modes):
                    ax = (
                        axes[d][subplot_index]
                        if separate_rows
                        else axes[0][subplot_index]
                    )
                    ax.semilogy(modes_dim, **props_dim[d], zorder=30 + d)
                    ax.scatter(
                        len(modes_dim) - 1,
                        modes_dim[-1],
                        marker=".",
                        color=props_dim[d].get("color", "black"),
                        zorder=30 + d,
                    )
                    max_error[subplot_index][d] = max(
                        max_error[subplot_index][d], all_modes[d][-1]
                    )

    if plot_over_time:
        max_num_modes = np.max(
            np.array(
                [
                    [len(modes) for modes in all_modes]
                    for subplot_index in range(num_cols)
                    for mode_time_series in all_mode_time_series[
                        subplot_index
                    ].values()
                    for _, all_modes in mode_time_series
                ]
            ),
            axis=0,
        )
        mode_cmap = [
            LinearSegmentedColormap.from_list(
                "Modes",
                ["black", props_dim[d].get("color", "black")],
                N=max_num_modes[d],
            )
            for d in range(len(monitor_labels))
        ]
        for subplot_index in range(num_cols):
            for element_id, mode_time_series in all_mode_time_series[
                subplot_index
            ].items():
                times = np.array([time for time, _ in mode_time_series])
                for d in range(len(monitor_labels)):
                    ax = axes[d][subplot_index]
                    for mode in range(max_num_modes[d]):
                        mode_time_series_i = np.array(
                            [
                                (
                                    all_modes[d][mode]
                                    if len(all_modes[d]) > mode
                                    else np.nan
                                )
                                for _, all_modes in mode_time_series
                            ]
                        )
                        color = mode_cmap[d](mode / (max_num_modes[d] - 1))
                        ax.semilogy(
                            times,
                            mode_time_series_i,
                            color=color,
                            zorder=30 + d,
                        )
        # Plot colorbars as legend
        import matplotlib.cm
        import matplotlib.colors

        for d in range(len(monitor_labels)):
            colorbar = plt.colorbar(
                matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(0, max_num_modes[d]),
                    cmap=mode_cmap[d],
                ),
                ax=axes[d],
                ticks=list(range(max_num_modes[d])),
                label=monitor_labels[d] + " Mode",
            )
            colorbar.ax.invert_yaxis()
    else:
        for subplot_index in range(num_cols):
            for d in range(len(monitor_labels)):
                ax = (
                    axes[d][subplot_index]
                    if separate_rows
                    else axes[0][subplot_index]
                )
                ax.axhline(
                    max_error[subplot_index][d], **props_dim[d], zorder=20 + d
                )
                ax.annotate(
                    monitor_labels[d],
                    xy=(0, max_error[subplot_index][d]),
                    xytext=((2 * d + 0.5) * plt.rcParams["font.size"], 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    bbox=dict(
                        fc="white",
                        ec=props_dim[d].get("color", "black"),
                        pad=2.0,
                    ),
                    zorder=40 + d,
                )

    # Set plot titles
    for subplot_index, ax in enumerate(axes[0]):
        ax.set_title(block_or_group_names[subplot_index], loc="left")
        num_elements_i = (
            len(all_mode_time_series[subplot_index])
            if plot_over_time
            else num_elements[subplot_index]
        )
        ax.set_title(
            f"{num_elements_i} element" + ("" if num_elements_i == 1 else "s"),
            loc="right",
        )

    for axes_row in axes:
        for ax in axes_row:
            # Draw grid lines
            ax.grid(which="both", zorder=0)
            # Allow only integer ticks for modes
            if not plot_over_time:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add y-labels to the leftmost subplots
    if plot_over_time:
        for d, ax in enumerate(axes):
            ax[0].set_ylabel(
                r"Power monitors $P_{q_" + monitor_labels[d].strip("$") + "}$"
            )
    elif separate_rows:
        for d, ax in enumerate(axes):
            ax[0].set_ylabel(
                r"Power monitors $P_{q_" + monitor_labels[d].strip("$") + "}$"
            )
    else:
        axes[0][0].set_ylabel(r"Power monitors $P_{q_{\hat{\imath}}}$")

    # Add x-label spanning all subplots
    ax_colspan = fig.add_subplot(111, frameon=False)
    ax_colspan.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_colspan.grid(False)
    ax_colspan.set_xlabel("Time" if plot_over_time else "Mode number")


@click.command(name="power-monitors")
@open_volfiles_command(
    obs_id_required=False, vars_required=False, multiple_vars=True
)
@click.option(
    "--list-blocks",
    is_flag=True,
    help="Print available blocks and block groups and exit.",
)
@click.option(
    "--block",
    "-b",
    "block_or_group_names",
    multiple=True,
    help=(
        "Name of block or block group to analyze. "
        "Can be specified multiple times to plot several block(groups) at once."
    ),
)
@click.option(
    "--elements",
    "-e",
    "element_patterns",
    multiple=True,
    help=(
        "Include only elements that match the specified glob "
        "pattern, like 'B*,(L1I*,L0I0,L0I0)'. "
        "Can be specified multiple times, in which case elements "
        "are included that match _any_ of the specified "
        "patterns. If unspecified, include all elements in the blocks."
    ),
)
@click.option(
    "--list-elements",
    is_flag=True,
    help=(
        "List all elements in the specified blocks subject to "
        "'--elements' / '-e' patterns."
    ),
)
@click.option(
    "--over-time", "-T", is_flag=True, help="Plot power monitors over time."
)
@click.option(
    "--tensor",
    "tensor_variable",
    help=(
        "Tensor variable to analyze as a whole, combining components "
        "internally. For shells this uses the shell-aware tensor path on "
        "radial-first shell meshes."
    ),
)
@click.option(
    "--skip-filtered-modes",
    type=int,
    default=0,
    help=(
        "Skip this number of highest modes. Useful if the highest modes are"
        " filtered, zeroing them out."
    ),
)
# Plotting options
@click.option("--figsize", nargs=2, type=float, help="Figure size in inches.")
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_power_monitors_command(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    vars,
    tensor_variable,
    list_blocks,
    block_or_group_names,
    list_elements,
    element_patterns,
    over_time,
    **kwargs,
):
    """Plot power monitors from volume data

    Reads volume data in the 'H5_FILES' and computes power monitors, which are
    essentially the spectral modes in each dimension of the grid. They give an
    indication how well the spectral expansion resolves fields on the grid.
    Component mode is selected with '--var' / '-y', while '--tensor' selects a
    tensor variable and combines its components internally.

    One subplot is created for every selected '--block' / '-b'. This can be a
    single block name, or a block group defined by the domain (such as all six
    wedges in a spherical shell). The power monitors in every logical direction
    of the grid are plotted for all elements in the block or block group. The
    logical directions are labeled "xi", "eta" and "zeta", and their orientation
    is defined by the coordinate maps in the domain. For example, see the
    documentation of the 'Wedge' map to understand which logical direction is
    radial in spherical shells. Shell-aware monitors currently support only
    radial-first shells, and plotting domains that mix spherical-shell elements
    with non-shell elements is not yet supported.
    """
    if over_time == (obs_id is not None):
        raise click.UsageError(
            "Specify an observation '--step' or '--time', or specify"
            " '--over-time' (but not both)."
        )
    if bool(vars) == bool(tensor_variable):
        raise click.UsageError(
            "Specify either '--var' / '-y' or '--tensor', but not both."
        )

    # Print available blocks and groups
    open_h5_file = spectre_h5.H5File(h5_files[0], "r")
    volfile = open_h5_file.get_vol(subfile_name)
    obs_id_for_components = obs_id or volfile.list_observation_ids()[0]
    all_tensor_components = volfile.list_tensor_components(
        obs_id_for_components
    )
    if tensor_variable is not None:
        vars = _select_tensor_components(all_tensor_components, tensor_variable)
    dim = volfile.get_dimension()
    domain = deserialize_domain[dim](volfile.get_domain())
    all_block_groups = list(domain.block_groups.keys())
    all_block_names = [block.name for block in domain.blocks]
    if list_blocks:
        import rich.columns

        rich.print(rich.columns.Columns(all_block_groups + all_block_names))
        return
    elif not block_or_group_names:
        raise RequiredChoiceError(
            (
                "Specify '--block' / '-b' to select (possibly multiple) blocks"
                " or block groups to analyze."
            ),
            choices=all_block_groups + all_block_names,
        )
    # Validate block and group names
    for name in block_or_group_names:
        if not (name in all_block_groups or name in all_block_names):
            raise RequiredChoiceError(
                f"'{name}' matches no block or block group.",
                choices=all_block_groups + all_block_names,
            )

    # Print available elements IDs
    if not element_patterns:
        # Don't apply any filters when no element patterns were specified
        element_patterns = None
    if list_elements:
        all_element_ids = sorted(
            set(
                element.id
                for element in iter_elements(
                    open_volfiles(h5_files, subfile_name, obs_id),
                    obs_id,
                    element_patterns=element_patterns,
                )
            )
        )
        # Print grouped by block
        import rich.console

        console = rich.console.Console()
        for i, block_name in enumerate(block_or_group_names):
            element_ids = [
                stripped_element_name(element_id)
                for element_id in all_element_ids
                if find_block_or_group(
                    element_id.block_id, block_or_group_names, domain
                )
                == i
            ]
            console.rule(
                f"[bold]{block_name}[/bold] ({len(element_ids)} elements)"
            )
            console.print(rich.columns.Columns(element_ids))
        return

    # Close the H5 file because we're done with preprocessing
    open_h5_file.close()

    # Plot!
    import rich.progress

    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        disable=(len(h5_files) == 1),
    )
    task_id = progress.add_task("Processing files", total=len(h5_files))
    volfiles_progress = progress.track(
        open_volfiles(h5_files, subfile_name, obs_id), task_id=task_id
    )
    with progress:
        plot_power_monitors(
            volfiles_progress,
            obs_id=obs_id,
            tensor_components=vars,
            tensor_variable=tensor_variable,
            domain=domain,
            block_or_group_names=block_or_group_names,
            element_patterns=element_patterns,
            **kwargs,
        )
        progress.update(task_id, completed=len(h5_files))


if __name__ == "__main__":
    plot_power_monitors_command(help_option_names=["-h", "--help"])
