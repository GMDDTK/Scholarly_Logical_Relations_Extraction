from __future__ import annotations

import hashlib
import html
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from .contracts.network_visualization import NetworkBundle


_ASSET_ROOT = Path(__file__).with_name("visualization_assets")
_ECHARTS_PATH = _ASSET_ROOT / "echarts.min.js"
_ECHARTS_SHA256 = "b66b25aeb4df84e33199dc21694014d336d222cbd9deb0e5a7c14bd6aa0d0fd0"
_MINIMUM_COMPLETE_HTML_BYTES = 100_000


@dataclass(frozen=True)
class InteractiveHtmlBuildResult:
    output_path: Path
    reused: bool


def render_interactive_html(
    *,
    bundle: NetworkBundle,
    output_path: Path,
    temp_root: Path,
) -> InteractiveHtmlBuildResult:
    if _html_matches(output_path, bundle.input_fingerprint):
        return InteractiveHtmlBuildResult(output_path=output_path, reused=True)

    runtime = _load_verified_echarts_runtime()
    data_json = _safe_embedded_json(bundle.model_dump_json())
    document = (
        _HTML_TEMPLATE.replace("__ECHARTS_RUNTIME__", runtime)
        .replace("__NETWORK_DATA__", data_json)
        .replace("__TOPIC_TEXT__", html.escape(bundle.topic, quote=True))
        .replace("__FINGERPRINT__", bundle.input_fingerprint)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temporary = temp_root / f"index.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(document, encoding="utf-8", newline="\n")
        if not _html_matches(temporary, bundle.input_fingerprint):
            raise RuntimeError("generated interactive HTML failed completeness checks")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return InteractiveHtmlBuildResult(output_path=output_path, reused=False)


def _load_verified_echarts_runtime() -> str:
    runtime_bytes = _ECHARTS_PATH.read_bytes()
    actual = hashlib.sha256(runtime_bytes).hexdigest()
    if actual != _ECHARTS_SHA256:
        raise RuntimeError(
            "vendored ECharts runtime checksum mismatch; "
            f"expected {_ECHARTS_SHA256}, got {actual}"
        )
    return runtime_bytes.decode("utf-8")


def _safe_embedded_json(value: str) -> str:
    return (
        value.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _html_matches(path: Path, fingerprint: str) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < _MINIMUM_COMPLETE_HTML_BYTES:
            return False
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return (
        f'data-visualization-fingerprint="{fingerprint}"' in content
        and "echarts.init" in content
        and 'id="network-data"' in content
        and content.rstrip().endswith("</html>")
    )


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN" data-visualization-fingerprint="__FINGERPRINT__">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="MA-LRNC reusable offline visualization">
  <title>__TOPIC_TEXT__ · 文献关系网络</title>
  <style>
    :root {
      --ink: #18212b; --muted: #637083; --paper: #f6f3ed; --panel: #fffdfa;
      --line: #d9d4ca; --accent: #006d77; --accent-soft: #d8eeec;
      --warning: #b95d18; --shadow: 0 12px 34px rgba(31, 43, 55, .09);
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--paper); font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
    header { padding: 30px 34px 18px; background: linear-gradient(135deg, #073b4c 0%, #006d77 68%, #2a9d8f 100%); color: white; }
    header .eyebrow { letter-spacing: .18em; font-size: 12px; opacity: .78; }
    h1 { margin: 8px 0 6px; font-size: clamp(25px, 3.2vw, 40px); font-weight: 700; }
    header p { margin: 0; max-width: 960px; opacity: .85; line-height: 1.7; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 12px; padding: 18px 34px 0; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); }
    .metric strong { display: block; font-size: 25px; color: var(--accent); }
    .metric span { color: var(--muted); font-size: 12px; }
    main { padding: 18px 34px 34px; }
    .toolbar { display: grid; grid-template-columns: 180px 170px minmax(220px, 1fr); gap: 12px; margin-bottom: 12px; }
    label.control { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    select, input { width: 100%; border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 9px 10px; color: var(--ink); }
    .relation-box { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; margin-bottom: 12px; }
    .relation-box > span { color: var(--muted); font-size: 12px; margin-right: 14px; }
    #relation-filters { display: inline-flex; flex-wrap: wrap; gap: 8px 14px; }
    #relation-filters label { font-size: 13px; white-space: nowrap; }
    #relation-filters input { width: auto; vertical-align: -1px; accent-color: var(--accent); }
    .grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(270px, .75fr); gap: 14px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow); overflow: hidden; }
    .card h2 { font-size: 15px; margin: 0; padding: 13px 16px 0; }
    .card .hint { padding: 3px 16px 0; color: var(--muted); font-size: 11px; }
    #network-chart { height: 590px; }
    #matrix-chart { height: min(720px, 72vw); min-height: 430px; }
    #detail-panel { min-height: 180px; padding: 14px 16px 18px; line-height: 1.65; overflow-wrap: anywhere; }
    #detail-panel h3 { margin: 0 0 8px; color: var(--accent); }
    #detail-panel h4 { margin: 13px 0 4px; font-size: 13px; }
    #detail-panel p, #detail-panel ul { margin: 4px 0; font-size: 13px; }
    #detail-panel ul { padding-left: 18px; }
    .review-list { padding: 12px 16px 18px; max-height: 300px; overflow: auto; }
    .review-item { border-left: 3px dashed var(--warning); padding: 7px 9px; margin: 7px 0; background: #fff5ea; font-size: 12px; }
    .full { grid-column: 1 / -1; }
    footer { padding: 16px 34px 28px; color: var(--muted); font-size: 11px; line-height: 1.7; }
    @media (max-width: 880px) {
      .summary { grid-template-columns: repeat(2, 1fr); } .toolbar { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; } #network-chart { height: 520px; }
    }
    @media print { header { background: #073b4c !important; print-color-adjust: exact; } .toolbar, .relation-box { display: none; } .grid { display: block; } .card { break-inside: avoid; margin-bottom: 12px; box-shadow: none; } }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">LITERATURE RELATIONSHIP ATLAS · OFFLINE</div>
    <h1>__TOPIC_TEXT__</h1>
    <p>严格网络仅纳入已接受关系；暂定网络会额外显示需人工复核的关系，并用虚线明确标记。所有视图共用同一份数据与固定坐标。</p>
  </header>
  <section class="summary" aria-label="数据概览">
    <div class="metric"><strong id="metric-papers">—</strong><span>论文</span></div>
    <div class="metric"><strong id="metric-strict">—</strong><span>严格关系边</span></div>
    <div class="metric"><strong id="metric-manual">—</strong><span>待人工复核结论</span></div>
    <div class="metric"><strong id="metric-years">—</strong><span>年份跨度</span></div>
  </section>
  <main>
    <section class="toolbar" aria-label="视图控制">
      <label class="control">网络口径
        <select id="projection-select"><option value="strict">严格网络（推荐）</option><option value="provisional">暂定网络（含待复核）</option></select>
      </label>
      <label class="control">发表年份
        <select id="year-select"><option value="all">全部年份</option></select>
      </label>
      <label class="control">检索论文
        <input id="paper-search" type="search" placeholder="输入论文编号、题名或作者">
      </label>
    </section>
    <section class="relation-box"><span>关系类型</span><div id="relation-filters"></div></section>
    <section class="grid">
      <article class="card">
        <h2>逻辑关系网络</h2><div class="hint">节点颜色表示年份，节点大小表示当前口径下的连接度；点击节点或边查看证据。</div>
        <div id="network-chart" role="img" aria-label="论文逻辑关系网络"></div>
      </article>
      <aside>
        <article class="card"><h2>论文与关系详情</h2><div id="detail-panel" aria-live="polite"></div></article>
        <article class="card" style="margin-top:14px"><h2>待人工复核</h2><div id="manual-review-list" class="review-list"></div></article>
      </aside>
      <article class="card full">
        <h2>全关系矩阵</h2><div class="hint">灰色为无实质关系；彩色为六类关系；橙色虚线边框表示仍需人工复核。</div>
        <div id="matrix-chart" role="img" aria-label="论文两两关系矩阵"></div>
      </article>
    </section>
  </main>
  <footer>
    数据指纹：__FINGERPRINT__<br>
    图表运行时：Apache ECharts 6.1.0，Apache License 2.0。该文件已内嵌运行时与数据，可离线双击打开；出版级 SVG/PDF 请使用同目录 figures 文件夹中的静态图件。
  </footer>

  <script id="network-data" type="application/json">__NETWORK_DATA__</script>
  <script>__ECHARTS_RUNTIME__</script>
  <script>
  "use strict";
  const DATA = JSON.parse(document.getElementById("network-data").textContent);
  const RELATIONS = ["承接", "拓展", "印证", "修正", "挑战", "互补"];
  const COLORS = {"承接":"#0072B2","拓展":"#E69F00","印证":"#009E73","修正":"#CC79A7","挑战":"#D55E00","互补":"#56B4E9"};
  const PROFILE_LABELS = {
    research_questions: "研究问题", theories_and_viewpoints: "理论与观点",
    methods_and_models: "方法与模型", data_and_contexts: "数据与情境",
    main_findings: "主要发现", limitations_and_boundary_conditions: "局限与边界条件"
  };
  const projectionSelect = document.getElementById("projection-select");
  const yearSelect = document.getElementById("year-select");
  const searchInput = document.getElementById("paper-search");
  const detailPanel = document.getElementById("detail-panel");
  const networkChart = echarts.init(document.getElementById("network-chart"), null, {renderer: "svg"});
  const matrixChart = echarts.init(document.getElementById("matrix-chart"), null, {renderer: "svg"});

  function textElement(tag, text, className) {
    const element = document.createElement(tag); element.textContent = text;
    if (className) element.className = className; return element;
  }
  function pairKey(record) { return record.paper_a_id + "__" + record.paper_b_id; }
  function edgeFor(outcome) {
    const record = outcome.relation_record; if (record.record_type !== "related") return null;
    let source = record.paper_a_id, target = record.paper_b_id;
    if (record.direction === "b_to_a") { source = record.paper_b_id; target = record.paper_a_id; }
    return { id: pairKey(record), source, target, relationType: record.relation_type,
      direction: record.direction, reviewStatus: outcome.review_status, outcome };
  }
  function selectedTypes() {
    return new Set(Array.from(document.querySelectorAll('#relation-filters input:checked')).map(x => x.value));
  }
  function visibleNodes() {
    const year = yearSelect.value;
    return DATA.nodes.filter(node => year === "all" || node.publication_date.slice(0,4) === year);
  }
  function visibleEdges(nodes) {
    const ids = new Set(nodes.map(node => node.paper_id)), types = selectedTypes();
    return DATA.pairs.map(edgeFor).filter(edge => edge && ids.has(edge.source) && ids.has(edge.target)
      && types.has(edge.relationType) && (projectionSelect.value === "provisional" || edge.reviewStatus === "accepted"));
  }
  function populateControls() {
    const filterRoot = document.getElementById("relation-filters");
    RELATIONS.forEach(type => { const label = document.createElement("label"), input = document.createElement("input");
      input.type = "checkbox"; input.value = type; input.checked = true; input.addEventListener("change", renderNetwork);
      label.append(input, document.createTextNode(" " + type)); filterRoot.appendChild(label); });
    const years = Array.from(new Set(DATA.nodes.map(node => node.publication_date.slice(0,4)))).sort();
    years.forEach(year => { const option = document.createElement("option"); option.value = year; option.textContent = year; yearSelect.appendChild(option); });
  }
  function renderMetrics() {
    const edges = DATA.pairs.map(edgeFor).filter(Boolean), years = DATA.nodes.map(n => Number(n.publication_date.slice(0,4)));
    document.getElementById("metric-papers").textContent = String(DATA.nodes.length);
    document.getElementById("metric-strict").textContent = String(edges.filter(e => e.reviewStatus === "accepted").length);
    document.getElementById("metric-manual").textContent = String(
      DATA.pairs.filter(o => o.review_status === "manual_review_required").length
    );
    document.getElementById("metric-years").textContent = Math.min(...years) + "–" + Math.max(...years);
  }
  function renderNetwork() {
    const nodes = visibleNodes(), edges = visibleEdges(nodes), degrees = Object.fromEntries(nodes.map(n => [n.paper_id, 0]));
    edges.forEach(edge => { degrees[edge.source]++; degrees[edge.target]++; });
    const query = searchInput.value.trim().toLowerCase();
    const graphNodes = nodes.map(node => { const pos = DATA.layout.positions[node.paper_id];
      const haystack = [node.paper_id, node.title, ...node.authors].join(" ").toLowerCase(), match = !query || haystack.includes(query);
      return { id: node.paper_id, name: node.paper_id, value: node.publication_date.slice(0,4), x: pos.x * 1000, y: -pos.y * 1000,
        symbolSize: 18 + Math.sqrt(degrees[node.paper_id] || 0) * 7, itemStyle: {opacity: match ? 1 : .18},
        label: {opacity: match ? 1 : .22}, rawNode: node };
    });
    const graphEdges = edges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target,
      symbol: edge.direction === "symmetric" ? ["none","none"] : ["none","arrow"], symbolSize: 8,
      lineStyle: {color: COLORS[edge.relationType], width: edge.reviewStatus === "accepted" ? 1.8 : 2.4,
        type: edge.reviewStatus === "accepted" ? "solid" : "dashed", opacity: .82, curveness: .06}, rawEdge: edge }));
    networkChart.setOption({ animation: false, tooltip: {trigger: "item", renderMode: "richText",
      formatter: p => p.dataType === "node" ? p.data.rawNode.paper_id + "\n" + p.data.rawNode.title : p.data.rawEdge.relationType + "\n" + p.data.rawEdge.id},
      visualMap: {type: "continuous", min: Math.min(...nodes.map(n => Number(n.publication_date.slice(0,4)))),
        max: Math.max(...nodes.map(n => Number(n.publication_date.slice(0,4)))), dimension: 0, show: true,
        orient: "horizontal", left: 16, bottom: 5, text: ["新", "早"], inRange: {color: ["#d8eeec", "#006d77", "#073b4c"]}},
      series: [{type: "graph", layout: "none", roam: true, data: graphNodes, edges: graphEdges,
        label: {show: true, position: "right", fontSize: 10}, emphasis: {focus: "adjacency"}, edgeSymbolSize: 8}]
    }, true);
  }
  function renderMatrix() {
    const ids = DATA.nodes.map(n => n.paper_id), index = Object.fromEntries(ids.map((id,i) => [id,i]));
    const values = DATA.pairs.map(outcome => { const r = outcome.relation_record;
      return {value: [index[r.paper_a_id], index[r.paper_b_id], r.record_type === "related" ? RELATIONS.indexOf(r.relation_type)+1 : 0],
        itemStyle: {borderColor: outcome.review_status === "manual_review_required" ? "#b95d18" : "#fffdfa",
          borderWidth: outcome.review_status === "manual_review_required" ? 2.5 : .8, borderType: outcome.review_status === "manual_review_required" ? "dashed" : "solid"}, outcome};
    });
    const mirrored = values.concat(values.map(item => ({...item, value: [item.value[1], item.value[0], item.value[2]]})));
    ids.forEach((id,i) => mirrored.push({value:[i,i,-1], itemStyle:{color:"#e8e4dc",borderColor:"#fffdfa",borderWidth:.8}}));
    matrixChart.setOption({animation:false, grid:{left:72,right:26,top:24,bottom:68}, tooltip:{trigger:"item",renderMode:"richText",
      formatter:p => { if (!p.data.outcome) return ids[p.value[0]]; const r=p.data.outcome.relation_record;
        return r.paper_a_id+" × "+r.paper_b_id+"\n"+(r.record_type==="related"?r.relation_type:"无实质关系")+"\n"+p.data.outcome.review_status; }},
      xAxis:{type:"category",data:ids,axisLabel:{rotate:60,fontSize:9},splitArea:{show:true}},
      yAxis:{type:"category",data:ids,axisLabel:{fontSize:9},splitArea:{show:true}},
      visualMap:{type:"piecewise",show:true,orient:"horizontal",left:"center",bottom:6,dimension:2,
        pieces:[{value:0,label:"无实质关系",color:"#d8d5cf"},...RELATIONS.map((r,i)=>({value:i+1,label:r,color:COLORS[r]}))]},
      series:[{type:"heatmap",data:mirrored,emphasis:{itemStyle:{shadowBlur:8,shadowColor:"rgba(0,0,0,.25)"}}}]
    }, true);
  }
  function clearDetail(message) { detailPanel.replaceChildren(textElement("p", message)); }
  function showNode(node) {
    detailPanel.replaceChildren(); detailPanel.append(textElement("h3", node.paper_id + " · " + node.title));
    detailPanel.append(textElement("p", node.authors.join("、") + " · " + node.publication_date));
    Object.entries(PROFILE_LABELS).forEach(([key,label]) => { const items=node.profile[key]; if (!items.length) return;
      detailPanel.append(textElement("h4", label)); const ul=document.createElement("ul");
      items.forEach(item => ul.append(textElement("li", item.statement))); detailPanel.append(ul); });
  }
  function showEdge(edge) {
    const outcome=edge.outcome, r=outcome.relation_record; detailPanel.replaceChildren();
    detailPanel.append(textElement("h3", edge.id + " · " + r.relation_type));
    detailPanel.append(textElement("p", outcome.review_status === "accepted" ? "已接受" : "需人工复核"));
    detailPanel.append(textElement("h4", "关系解释"), textElement("p", r.relation_explanation));
    detailPanel.append(textElement("h4", r.paper_a_id + " 证据")); const a=document.createElement("ul");
    r.paper_a_evidence_quotes.forEach(q=>a.append(textElement("li",q))); detailPanel.append(a);
    detailPanel.append(textElement("h4", r.paper_b_id + " 证据")); const b=document.createElement("ul");
    r.paper_b_evidence_quotes.forEach(q=>b.append(textElement("li",q))); detailPanel.append(b);
  }
  function renderManualReviewList() {
    const root=document.getElementById("manual-review-list"); root.replaceChildren();
    const pending=DATA.pairs.filter(o=>o.review_status==="manual_review_required");
    if (!pending.length) { root.append(textElement("p", "无待人工复核关系。")); return; }
    pending.forEach(outcome => { const r=outcome.relation_record, item=textElement("div", pairKey(r)+" · "+(r.relation_type||"无实质关系"),"review-item");
      item.tabIndex=0; item.addEventListener("click",()=>{const edge=edgeFor(outcome); if(edge) showEdge(edge);}); root.append(item); });
  }
  populateControls(); renderMetrics(); renderNetwork(); renderMatrix(); renderManualReviewList();
  clearDetail("点击网络中的论文节点或关系边查看画像、证据与审核状态。");
  projectionSelect.addEventListener("change", renderNetwork); yearSelect.addEventListener("change", renderNetwork);
  searchInput.addEventListener("input", renderNetwork);
  networkChart.on("click", p => { if (p.dataType === "node") showNode(p.data.rawNode); else if (p.dataType === "edge") showEdge(p.data.rawEdge); });
  matrixChart.on("click", p => { if (!p.data.outcome) return; const edge=edgeFor(p.data.outcome);
    if(edge) showEdge(edge); else { const r=p.data.outcome.relation_record; clearDetail(r.paper_a_id+" × "+r.paper_b_id+"："+r.no_relation_reason); } });
  window.addEventListener("resize", () => { networkChart.resize(); matrixChart.resize(); });
  </script>
</body>
</html>'''
