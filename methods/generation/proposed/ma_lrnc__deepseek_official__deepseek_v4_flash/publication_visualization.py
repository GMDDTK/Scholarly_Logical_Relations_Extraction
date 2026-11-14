from __future__ import annotations

import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_PROJECT_ROOT / "tmp" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from pypdf import PdfReader

from .contracts.network_visualization import NetworkBundle, NetworkEdgeView
from .contracts.relation_record import RelationType
from .network_visualization import build_projection_edges


_FIGURE_STEMS = (
    "strict_network",
    "relation_matrix",
    "relation_type_distribution",
)
_FORMATS = ("svg", "pdf")
_FONT_PREFERENCES = (
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
)
_RELATION_ORDER = (
    RelationType.CARRIES_FORWARD,
    RelationType.EXTENDS,
    RelationType.CORROBORATES,
    RelationType.REVISES,
    RelationType.CHALLENGES,
    RelationType.COMPLEMENTS,
)
_RELATION_COLORS = {
    RelationType.CARRIES_FORWARD: "#0072B2",
    RelationType.EXTENDS: "#E69F00",
    RelationType.CORROBORATES: "#009E73",
    RelationType.REVISES: "#CC79A7",
    RelationType.CHALLENGES: "#D55E00",
    RelationType.COMPLEMENTS: "#56B4E9",
}
_RELATION_LINESTYLES = {
    RelationType.CARRIES_FORWARD: "solid",
    RelationType.EXTENDS: (0, (5, 2)),
    RelationType.CORROBORATES: (0, (1, 1)),
    RelationType.REVISES: (0, (4, 1, 1, 1)),
    RelationType.CHALLENGES: (0, (2, 1)),
    RelationType.COMPLEMENTS: (0, (6, 2, 1, 2)),
}


@dataclass(frozen=True)
class PublicationFigureBuildResult:
    output_paths: tuple[Path, ...]
    reused: bool


def render_publication_figures(
    *,
    bundle: NetworkBundle,
    output_root: Path,
    temp_root: Path,
) -> PublicationFigureBuildResult:
    output_paths = tuple(
        output_root / f"{stem}.{suffix}"
        for stem in _FIGURE_STEMS
        for suffix in _FORMATS
    )
    if _complete_set_matches(
        output_paths=output_paths,
        fingerprint=bundle.input_fingerprint,
    ):
        return PublicationFigureBuildResult(
            output_paths=output_paths,
            reused=True,
        )

    font_family = _select_chinese_font()
    temp_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    figures: dict[str, plt.Figure] = {}
    temporary_paths: dict[Path, Path] = {}
    rc = {
        "font.family": font_family,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": bundle.input_fingerprint,
    }
    try:
        with matplotlib.rc_context(rc):
            figures = {
                "strict_network": _create_strict_network_figure(
                    bundle=bundle,
                    font_family=font_family,
                ),
                "relation_matrix": _create_relation_matrix_figure(
                    bundle=bundle,
                    font_family=font_family,
                ),
                "relation_type_distribution": _create_distribution_figure(
                    bundle=bundle,
                    font_family=font_family,
                ),
            }
            for stem, figure in figures.items():
                for suffix in _FORMATS:
                    destination = output_root / f"{stem}.{suffix}"
                    temporary = (
                        temp_root
                        / f"{stem}.{uuid.uuid4().hex}.tmp.{suffix}"
                    )
                    _save_figure(
                        figure=figure,
                        destination=temporary,
                        suffix=suffix,
                        bundle=bundle,
                        title=_figure_title(stem=stem, topic=bundle.topic),
                    )
                    temporary_paths[destination] = temporary
        for destination, temporary in temporary_paths.items():
            os.replace(temporary, destination)
    finally:
        for figure in figures.values():
            plt.close(figure)
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)

    if not _complete_set_matches(
        output_paths=output_paths,
        fingerprint=bundle.input_fingerprint,
    ):
        raise ValueError("publication figure set failed post-write validation")
    return PublicationFigureBuildResult(
        output_paths=output_paths,
        reused=False,
    )


def _create_strict_network_figure(
    *,
    bundle: NetworkBundle,
    font_family: str,
):
    edges = build_projection_edges(bundle, projection="strict")
    positions = {
        paper_id: (position.x, position.y)
        for paper_id, position in bundle.layout.positions.items()
    }
    graph = nx.Graph()
    graph.add_nodes_from(node.paper_id for node in bundle.nodes)
    graph.add_edges_from((edge.source, edge.target) for edge in edges)
    years = {node.paper_id: int(node.publication_date[:4]) for node in bundle.nodes}
    year_values = list(years.values())
    year_min = min(year_values)
    year_max = max(year_values)
    norm = Normalize(
        vmin=year_min - 0.5 if year_min == year_max else year_min,
        vmax=year_max + 0.5 if year_min == year_max else year_max,
    )
    cmap = plt.get_cmap("viridis")

    figure, axis = plt.subplots(figsize=(13.5, 10.0), constrained_layout=True)
    axis.scatter(
        [positions[node.paper_id][0] for node in bundle.nodes],
        [positions[node.paper_id][1] for node in bundle.nodes],
        s=[120 + 20 * graph.degree(node.paper_id) for node in bundle.nodes],
        c=[years[node.paper_id] for node in bundle.nodes],
        cmap=cmap,
        norm=norm,
        edgecolors="#1F2937",
        linewidths=0.6,
        zorder=3,
    )
    for edge in edges:
        _draw_publication_edge(axis=axis, positions=positions, edge=edge)
    for node in bundle.nodes:
        x, y = positions[node.paper_id]
        axis.text(
            x,
            y,
            node.paper_id,
            ha="center",
            va="center",
            fontsize=6.2,
            fontfamily=font_family,
            color="#111827",
            zorder=4,
        )

    relation_handles = [
        Line2D(
            [0],
            [0],
            color=_RELATION_COLORS[relation_type],
            linestyle=_RELATION_LINESTYLES[relation_type],
            linewidth=2.1,
            label=relation_type.value,
        )
        for relation_type in _RELATION_ORDER
    ]
    axis.legend(
        handles=relation_handles,
        title="关系类型",
        loc="upper left",
        frameon=True,
        ncols=2,
        prop={"family": font_family, "size": 8},
        title_fontproperties={"family": font_family, "size": 9},
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axis, fraction=0.032, pad=0.015)
    colorbar.set_label("发表年份", fontfamily=font_family)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title(
        f"{bundle.topic}：严格学术逻辑关系网络",
        fontsize=16,
        fontweight="bold",
        fontfamily=font_family,
        pad=14,
    )
    axis.text(
        0.5,
        -0.02,
        "仅包含 accepted + related；节点大小按严格网络度数，节点颜色按发表年份",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        fontfamily=font_family,
        color="#374151",
    )
    axis.set_axis_off()
    axis.margins(0.12)
    return figure


def _draw_publication_edge(
    *,
    axis,
    positions: dict[str, tuple[float, float]],
    edge: NetworkEdgeView,
) -> None:
    graph = nx.DiGraph() if edge.directed else nx.Graph()
    graph.add_edge(edge.source, edge.target)
    drawing_options = {
        "ax": axis,
        "edgelist": [(edge.source, edge.target)],
        "edge_color": _RELATION_COLORS[edge.relation_type],
        "style": _RELATION_LINESTYLES[edge.relation_type],
        "width": 1.15,
        "alpha": 0.68,
        "node_size": 190,
    }
    if edge.directed:
        drawing_options.update(
            {
                "arrows": True,
                "arrowstyle": "-|>",
                "arrowsize": 9,
                "connectionstyle": "arc3,rad=0.035",
                "min_source_margin": 3,
                "min_target_margin": 5,
            }
        )
    else:
        drawing_options["arrows"] = False
    nx.draw_networkx_edges(graph, positions, **drawing_options)


def _create_relation_matrix_figure(
    *,
    bundle: NetworkBundle,
    font_family: str,
):
    node_ids = [node.paper_id for node in bundle.nodes]
    index = {paper_id: position for position, paper_id in enumerate(node_ids)}
    code_by_relation = {
        relation_type: offset + 2
        for offset, relation_type in enumerate(_RELATION_ORDER)
    }
    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=int)
    manual_cells: list[tuple[int, int]] = []
    for outcome in bundle.pairs:
        record = outcome.relation_record
        row = index[record.paper_a_id]
        column = index[record.paper_b_id]
        code = (
            1
            if not hasattr(record, "relation_type")
            else code_by_relation[record.relation_type]
        )
        matrix[row, column] = code
        matrix[column, row] = code
        if outcome.review_status == "manual_review_required":
            manual_cells.extend(((row, column), (column, row)))

    colors = [
        "#FFFFFF",
        "#E5E7EB",
        *[_RELATION_COLORS[relation_type] for relation_type in _RELATION_ORDER],
    ]
    size = min(18.0, max(8.0, 4.0 + 0.24 * len(node_ids)))
    figure, axis = plt.subplots(
        figsize=(size, size),
        constrained_layout=True,
    )
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)
    axis.imshow(matrix, cmap=cmap, norm=norm, interpolation="none")
    for row, column in manual_cells:
        axis.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#111827",
                linewidth=1.2,
                linestyle=(0, (2, 1)),
            )
        )
    axis.set_xticks(range(len(node_ids)), labels=node_ids)
    axis.set_yticks(range(len(node_ids)), labels=node_ids)
    axis.tick_params(axis="x", labelrotation=90, labelsize=5.5)
    axis.tick_params(axis="y", labelsize=5.5)
    axis.set_xticks(np.arange(-0.5, len(node_ids), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(node_ids), 1), minor=True)
    axis.grid(which="minor", color="#FFFFFF", linewidth=0.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title(
        f"{bundle.topic}：全部论文对关系矩阵",
        fontsize=15,
        fontweight="bold",
        fontfamily=font_family,
        pad=12,
    )
    handles = [Patch(facecolor="#E5E7EB", label="无关系")]
    handles.extend(
        Patch(
            facecolor=_RELATION_COLORS[relation_type],
            label=relation_type.value,
        )
        for relation_type in _RELATION_ORDER
    )
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="#111827",
            linestyle=(0, (2, 1)),
            label="待人工复核（虚线框）",
        )
    )
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncols=4,
        frameon=True,
        prop={"family": font_family, "size": 7.5},
    )
    return figure


def _create_distribution_figure(
    *,
    bundle: NetworkBundle,
    font_family: str,
):
    edges = build_projection_edges(bundle, projection="strict")
    counts = Counter(edge.relation_type for edge in edges)
    labels = [relation_type.value for relation_type in _RELATION_ORDER]
    values = [counts[relation_type] for relation_type in _RELATION_ORDER]
    colors = [_RELATION_COLORS[relation_type] for relation_type in _RELATION_ORDER]
    figure, axis = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors, edgecolor="#1F2937", linewidth=0.6)
    axis.bar_label(bars, labels=[str(value) for value in values], padding=3)
    axis.set_title(
        f"{bundle.topic}：严格网络关系类型分布",
        fontsize=15,
        fontweight="bold",
        fontfamily=font_family,
        pad=12,
    )
    axis.set_xlabel("关系类型", fontfamily=font_family)
    axis.set_ylabel("已接受关系边数量", fontfamily=font_family)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def _select_chinese_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in _FONT_PREFERENCES:
        if font_name in available:
            return font_name
    raise ValueError(
        "No supported Chinese font found; expected one of: "
        + ", ".join(_FONT_PREFERENCES)
    )


def _save_figure(
    *,
    figure,
    destination: Path,
    suffix: str,
    bundle: NetworkBundle,
    title: str,
) -> None:
    if suffix == "svg":
        metadata = {
            "Title": title,
            "Description": bundle.input_fingerprint,
            "Creator": "MA-LRNC visualization",
            "Date": None,
        }
    else:
        metadata = {
            "Title": title,
            "Subject": bundle.input_fingerprint,
            "Creator": "MA-LRNC visualization",
            "CreationDate": None,
            "ModDate": None,
        }
    figure.savefig(
        destination,
        format=suffix,
        bbox_inches="tight",
        pad_inches=0.16,
        metadata=metadata,
    )


def _complete_set_matches(
    *,
    output_paths: tuple[Path, ...],
    fingerprint: str,
) -> bool:
    for path in output_paths:
        try:
            if path.stat().st_size <= 500:
                return False
            if path.suffix == ".svg":
                content = path.read_text(encoding="utf-8")
                if "<svg" not in content or fingerprint not in content:
                    return False
            else:
                reader = PdfReader(path)
                if len(reader.pages) != 1 or reader.metadata is None:
                    return False
                if fingerprint not in (reader.metadata.subject or ""):
                    return False
        except (OSError, UnicodeError, ValueError):
            return False
    return True


def _figure_title(*, stem: str, topic: str) -> str:
    titles = {
        "strict_network": "严格学术逻辑关系网络",
        "relation_matrix": "全部论文对关系矩阵",
        "relation_type_distribution": "严格网络关系类型分布",
    }
    return f"{topic}：{titles[stem]}"
