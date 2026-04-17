from __future__ import annotations


def clamp_value(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range.

    Args:
        value (float): Numeric value to clamp.
        minimum (float): Inclusive lower bound.
        maximum (float): Inclusive upper bound.

    Returns:
        float: Clamped numeric value.
    """
    return max(minimum, min(value, maximum))


def interpolate_range(
    value: float,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
) -> float:
    """Linearly interpolate a value between two numeric ranges.

    Args:
        value (float): Input value to interpolate.
        input_range (tuple[float, float]): Inclusive lower and upper bounds of the input scale.
        output_range (tuple[float, float]): Lower and upper bounds of the output scale.

    Returns:
        float: Interpolated value mapped onto the output range.
    """
    input_min, input_max = input_range
    output_min, output_max = output_range

    if input_max == input_min:
        return output_max

    clamped_value = clamp_value(value=value, minimum=input_min, maximum=input_max)
    scale = (clamped_value - input_min) / (input_max - input_min)
    return output_min + scale * (output_max - output_min)


def compute_clustermap_layout(
    pathway_count: int,
    gene_count: int,
    clustermap_cfg: dict,
) -> dict[str, float | tuple[float, float]]:
    """Compute clustermap figure and font settings from retained matrix dimensions.

    Args:
        pathway_count (int): Number of pathway rows retained in the clustermap.
        gene_count (int): Number of gene columns retained in the clustermap.
        clustermap_cfg (dict): Clustermap plotting configuration from the YAML file.

    Returns:
        dict[str, float | tuple[float, float]]: Figure size and font sizes derived from the retained matrix shape.
    """
    min_width, min_height = tuple(clustermap_cfg["figsize_min"])
    max_width, max_height = tuple(clustermap_cfg["figsize_max"])
    gene_count_range = tuple(clustermap_cfg["gene_count_range"])
    pathway_count_range = tuple(clustermap_cfg["pathway_count_range"])

    width = interpolate_range(
        value=float(gene_count),
        input_range=gene_count_range,
        output_range=(float(min_width), float(max_width)),
    )
    height = interpolate_range(
        value=float(pathway_count),
        input_range=pathway_count_range,
        output_range=(float(min_height), float(max_height)),
    )

    width_ratio = interpolate_range(
        value=width,
        input_range=(float(min_width), float(max_width)),
        output_range=(0.0, 1.0),
    )
    height_ratio = interpolate_range(
        value=height,
        input_range=(float(min_height), float(max_height)),
        output_range=(0.0, 1.0),
    )
    axis_ratio = max(width_ratio, height_ratio)
    label_ratio = (width_ratio + height_ratio) / 2.0
    x_tick_fontsize = interpolate_range(
        value=axis_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["x_tick_fontsize_min"]),
            float(clustermap_cfg["x_tick_fontsize_max"]),
        ),
    )
    x_label_fontsize = interpolate_range(
        value=label_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["x_label_fontsize_min"]),
            float(clustermap_cfg["x_label_fontsize_max"]),
        ),
    )
    y_tick_fontsize = interpolate_range(
        value=axis_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["y_tick_fontsize_min"]),
            float(clustermap_cfg["y_tick_fontsize_max"]),
        ),
    )
    y_label_fontsize = interpolate_range(
        value=label_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["y_label_fontsize_min"]),
            float(clustermap_cfg["y_label_fontsize_max"]),
        ),
    )
    cbar_label_fontsize = interpolate_range(
        value=label_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["cbar_label_fontsize_min"]),
            float(clustermap_cfg["cbar_label_fontsize_max"]),
        ),
    )
    cbar_tick_fontsize = interpolate_range(
        value=height_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["cbar_tick_fontsize_min"]),
            float(clustermap_cfg["cbar_tick_fontsize_max"]),
        ),
    )
    genes_dendrogram_ratio = interpolate_range(
        value=width_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["dendrogram_ratio_genes_min"]),
            float(clustermap_cfg["dendrogram_ratio_genes_max"]),
        ),
    )
    pathways_dendrogram_ratio = interpolate_range(
        value=height_ratio,
        input_range=(0.0, 1.0),
        output_range=(
            float(clustermap_cfg["dendrogram_ratio_pathways_min"]),
            float(clustermap_cfg["dendrogram_ratio_pathways_max"]),
        ),
    )

    return {
        "figsize": (width, height),
        "x_label_fontsize": x_label_fontsize,
        "x_tick_fontsize": x_tick_fontsize,
        "y_label_fontsize": y_label_fontsize,
        "y_tick_fontsize": y_tick_fontsize,
        "cbar_label_fontsize": cbar_label_fontsize,
        "cbar_tick_fontsize": cbar_tick_fontsize,
        "dendrogram_ratio": (pathways_dendrogram_ratio, genes_dendrogram_ratio),
    }


def truncate_display_label(label: str, max_length: int) -> str:
    """Truncate a display label and append an ellipsis when needed.

    Args:
        label (str): Original label text to display.
        max_length (int): Maximum number of characters to keep before appending an ellipsis.

    Returns:
        str: Truncated label text suitable for plotting.
    """
    if len(label) <= max_length:
        return label
    return f"{label[:max_length]}..."
