from __future__ import annotations

import html
import json
import math
from pathlib import Path

from .checkpointing import (
    compute_fingerprint,
    load_valid_outcome,
    write_text_atomically,
)
from .contracts import (
    CitationNetworkBundle,
    CitationNetworkEdge,
    CitationNetworkNode,
    TopicInputSet,
)
from .result_paths import CitationResultPaths
from .workflow import paper_input_fingerprint


class IncompleteTopicError(RuntimeError):
    pass


NODE_RADIUS = 14.0
EDGE_NODE_CLEARANCE = 3.0


def build_network_bundle(
    *,
    topic_input: TopicInputSet,
    paths: CitationResultPaths,
    model: str,
) -> CitationNetworkBundle:
    outcomes = []
    missing = []
    for document in topic_input.reference_documents:
        fingerprint = paper_input_fingerprint(
            topic_input=topic_input,
            document=document,
            model=model,
        )
        outcome = load_valid_outcome(
            path=paths.outcome(document.paper.paper_id),
            topic=topic_input.topic,
            paper_id=document.paper.paper_id,
            input_fingerprint=fingerprint,
        )
        if outcome is None:
            missing.append(document.paper.paper_id)
        else:
            outcomes.append(outcome)
    if missing:
        raise IncompleteTopicError(
            "Cannot build citation network; missing valid outcomes: "
            + ", ".join(missing)
        )

    nodes = []
    for index, paper in enumerate(topic_input.papers):
        angle = (2 * math.pi * index / len(topic_input.papers)) - (math.pi / 2)
        nodes.append(
            CitationNetworkNode(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=paper.authors,
                publication_date=paper.publication_date,
                x=round(math.cos(angle), 8),
                y=round(math.sin(angle), 8),
            )
        )
    edges = [
        CitationNetworkEdge(
            source_paper_id=outcome.paper_id,
            target_paper_id=cited.cited_paper_id,
            evidence=cited.evidence,
        )
        for outcome in outcomes
        for cited in outcome.cited_papers
    ]
    edges.sort(key=lambda edge: (edge.source_paper_id, edge.target_paper_id))
    papers_without_references = sorted(
        outcome.paper_id for outcome in outcomes if not outcome.has_references
    )
    bundle_fingerprint = compute_fingerprint(
        {
            "bundle_version": 1,
            "topic": topic_input.topic,
            "catalog_sha256": topic_input.catalog_sha256,
            "model": model,
            "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
        }
    )
    return CitationNetworkBundle(
        topic=topic_input.topic,
        input_fingerprint=bundle_fingerprint,
        nodes=nodes,
        edges=edges,
        papers_without_references=papers_without_references,
    )


def write_network_artifacts(
    *,
    bundle: CitationNetworkBundle,
    paths: CitationResultPaths,
) -> None:
    network_root = paths.network_root
    figures_root = network_root / "figures"
    network_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    write_text_atomically(
        text=bundle.model_dump_json(indent=2),
        destination=network_root / "network_data.json",
        temp_root=paths.temp_root,
        business_key="network_data",
    )
    write_text_atomically(
        text=_render_html(bundle),
        destination=network_root / "index.html",
        temp_root=paths.temp_root,
        business_key="network_index",
    )
    _render_publication_figure(
        bundle=bundle,
        svg_path=figures_root / "citation_network.svg",
        pdf_path=figures_root / "citation_network.pdf",
    )


def _render_html(bundle: CitationNetworkBundle) -> str:
    width = 1000
    height = 760
    radius = 310
    center_x = width / 2
    center_y = height / 2
    node_by_id = {node.paper_id: node for node in bundle.nodes}
    edge_elements = []
    for edge in bundle.edges:
        source = node_by_id[edge.source_paper_id]
        target = node_by_id[edge.target_paper_id]
        source_x, source_y = (
            center_x + radius * source.x,
            center_y + radius * source.y,
        )
        target_x, target_y = (
            center_x + radius * target.x,
            center_y + radius * target.y,
        )
        x1, y1, x2, y2 = _visible_edge_endpoints(
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            target_y=target_y,
        )
        edge_elements.append(
            f'<g class="citation-edge" data-source="{edge.source_paper_id}" '
            f'data-target="{edge.target_paper_id}" tabindex="0">'
            f'<title>{edge.source_paper_id} 引用 {edge.target_paper_id}</title>'
            f'<line class="edge-hit" x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" />'
            f'<line class="edge-line" x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" marker-end="url(#arrow)" />'
            "</g>"
        )
    node_elements = []
    for node in bundle.nodes:
        x, y = center_x + radius * node.x, center_y + radius * node.y
        metadata = html.escape(
            f"{node.paper_id} {node.title} | {'; '.join(node.authors)} | "
            f"{node.publication_date.isoformat()}"
        )
        node_elements.append(
            f'<g class="citation-node" data-paper-id="{node.paper_id}">'
            f'<title>{metadata}</title>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{NODE_RADIUS:g}" />'
            f'<text x="{x:.2f}" y="{y + 4:.2f}" text-anchor="middle" '
            f'font-size="8">{node.paper_id[1:]}</text></g>'
        )
    edge_rows = []
    for edge in bundle.edges:
        evidence = "<br>".join(
            f"[{item.row_number}] {html.escape(item.raw_text)}"
            for item in edge.evidence
        )
        edge_rows.append(
            "<tr>"
            f"<td>{edge.source_paper_id}</td>"
            f"<td>{edge.target_paper_id}</td>"
            f"<td>{evidence}</td>"
            "</tr>"
        )
    data = bundle.model_dump_json().replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(bundle.topic)}引用网络</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:24px;color:#0f172a}}
svg{{max-width:100%;height:auto;border:1px solid #e2e8f0;background:#fff}} .meta{{color:#475569}}
.edge-line{{stroke:#64748b;stroke-width:1.7;vector-effect:non-scaling-stroke;transition:stroke .15s,stroke-width .15s}}
.edge-hit{{stroke:transparent;stroke-width:14;vector-effect:non-scaling-stroke;pointer-events:stroke}}
.citation-edge:hover .edge-line,.citation-edge:focus .edge-line{{stroke:#dc2626;stroke-width:3}}
.citation-edge:focus{{outline:none}} .citation-node circle{{fill:#0f766e;stroke:#fff;stroke-width:1.5;transition:fill .15s,stroke .15s,stroke-width .15s}}
.citation-node text{{fill:#fff;pointer-events:none}} .citation-node.is-source circle{{fill:#d97706;stroke:#92400e;stroke-width:3}}
.citation-node.is-target circle{{fill:#dc2626;stroke:#991b1b;stroke-width:3}} .legend text{{font-size:15px;fill:#334155;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}}</style></head>
<body><h1>{html.escape(bundle.topic)}：主题内直接引用网络</h1>
<p class="meta">50 个固定节点，{len(bundle.edges)} 条有向引用边。悬停或聚焦连线可高亮引用双方。</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="主题内引用网络">
<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L0,10 L11,5 z" fill="#475569"/></marker></defs>
<g class="legend"><line class="edge-line" x1="32" y1="35" x2="92" y2="35" marker-end="url(#arrow)"/><text x="108" y="40">引用论文 → 被引论文</text></g>
{''.join(edge_elements)}{''.join(node_elements)}</svg>
<h2>引用边与原始证据</h2><table><thead><tr><th>引用论文</th><th>被引论文</th><th>参考文献证据</th></tr></thead>
<tbody>{''.join(edge_rows)}</tbody></table>
<script type="application/json" id="network-data">{data}</script>
<script>const nodes=new Map([...document.querySelectorAll('.citation-node')].map(node=>[node.dataset.paperId,node]));
for(const edge of document.querySelectorAll('.citation-edge')){{const toggle=active=>{{nodes.get(edge.dataset.source)?.classList.toggle('is-source',active);nodes.get(edge.dataset.target)?.classList.toggle('is-target',active)}};edge.addEventListener('mouseenter',()=>toggle(true));edge.addEventListener('mouseleave',()=>toggle(false));edge.addEventListener('focus',()=>toggle(true));edge.addEventListener('blur',()=>toggle(false));}}</script></body></html>"""


def _visible_edge_endpoints(
    *,
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
) -> tuple[float, float, float, float]:
    delta_x = target_x - source_x
    delta_y = target_y - source_y
    length = math.hypot(delta_x, delta_y)
    if length == 0:
        raise ValueError("citation edge endpoints must not share one position")
    unit_x = delta_x / length
    unit_y = delta_y / length
    offset = NODE_RADIUS + EDGE_NODE_CLEARANCE
    return (
        source_x + unit_x * offset,
        source_y + unit_y * offset,
        target_x - unit_x * offset,
        target_y - unit_y * offset,
    )


def _render_publication_figure(
    *,
    bundle: CitationNetworkBundle,
    svg_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (
            name
            for name in (
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "SimHei",
            )
            if name in available_fonts
        ),
        None,
    )
    if font_family is None:
        raise ValueError("No supported Chinese font is available for publication figures")

    with matplotlib.rc_context(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": bundle.input_fingerprint,
        }
    ):
        graph = nx.DiGraph()
        graph.add_nodes_from(node.paper_id for node in bundle.nodes)
        graph.add_edges_from(
            (edge.source_paper_id, edge.target_paper_id) for edge in bundle.edges
        )
        positions = {node.paper_id: (node.x, node.y) for node in bundle.nodes}
        figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edge_color="#64748b",
            arrows=True,
            arrowsize=18,
            width=1.2,
            min_source_margin=10,
            min_target_margin=14,
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_color="#0f766e",
            node_size=260,
            linewidths=0.5,
            edgecolors="white",
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node.paper_id: node.paper_id[1:] for node in bundle.nodes},
            font_size=6,
            font_color="white",
            font_family=font_family,
            ax=axis,
        )
        axis.set_title(
            f"{bundle.topic}：主题内直接引用网络（n=50, m={len(bundle.edges)}）",
            fontfamily=font_family,
        )
        axis.text(
            0.02,
            0.98,
            "方向：引用论文 → 被引论文",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#334155",
            fontsize=9,
            fontfamily=font_family,
        )
        axis.set_axis_off()
        figure.savefig(svg_path, format="svg", bbox_inches="tight")
        figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(figure)
