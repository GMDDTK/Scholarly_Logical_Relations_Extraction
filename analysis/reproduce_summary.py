"""Reconstruct the study's network summaries from the released workbooks."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import networkx as nx
from openpyxl import load_workbook

DATA = Path(__file__).resolve().parent.parent / "data"
CORPUS_FILE = "ESM_1_Publication_Dataset.xlsx"
RELATIONS_FILE = "ESM_2_Citation_and_Logical_Relations.xlsx"


def records(workbook, sheet):
    rows = workbook[sheet].values
    next(rows)  # Worksheet title
    headers = next(rows)
    return [dict(zip(headers, row, strict=True)) for row in rows]


def graph_summary(graph):
    components = sorted(map(len, nx.connected_components(graph)), reverse=True)
    return {
        "edges": graph.number_of_edges(),
        "isolates": nx.number_of_isolates(graph),
        "components": len(components),
        "largest_component": components[0],
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconstruct(topic, papers, citations, relations, decisions):
    paper_ids = {item["Topic paper ID"] for item in papers if item["Topic code"] == topic}
    expected_pairs = set(combinations(sorted(paper_ids), 2))
    by_pair = {(row["Paper A ID"], row["Paper B ID"]): row for row in decisions}
    require(set(by_pair) == expected_pairs and len(by_pair) == len(decisions), f"{topic}: incomplete pair decisions")

    directed_citations = nx.DiGraph()
    directed_citations.add_nodes_from(paper_ids)
    directed_citations.add_edges_from((row["Citing paper ID"], row["Cited paper ID"]) for row in citations)
    require(set(directed_citations) == paper_ids and nx.number_of_selfloops(directed_citations) == 0
            and directed_citations.number_of_edges() == len(citations), f"{topic}: invalid citation edges")
    citation_graph = directed_citations.to_undirected()
    logical_graph = nx.Graph()
    logical_graph.add_nodes_from(paper_ids)
    logical_graph.add_edges_from((row["Paper A ID"], row["Paper B ID"]) for row in relations)
    require(set(logical_graph) == paper_ids and nx.number_of_selfloops(logical_graph) == 0
            and logical_graph.number_of_edges() == len(relations), f"{topic}: invalid logical edges")

    shortest_paths = dict(nx.all_pairs_shortest_path_length(citation_graph))
    path_counts = Counter()
    relation_types = Counter()
    cited_relation_types = Counter()
    for relation in relations:
        a, b = relation["Paper A ID"], relation["Paper B ID"]
        label = relation["Principal label"]
        length = shortest_paths[a].get(b)
        path_counts[str(length) if length is not None else "no citation path"] += 1
        relation_types[label] += 1
        if length == 1:
            cited_relation_types[label] += 1

    for (a, b), row in by_pair.items():
        require(
            (row["Logical relation"] == "Yes") == logical_graph.has_edge(a, b)
            and (row["Recorded direct citation"] == "Yes") == citation_graph.has_edge(a, b)
            and row["Shortest citation path"] == shortest_paths[a].get(b),
            f"{topic}: inconsistent record for {a} and {b}",
        )

    logical_count = logical_graph.number_of_edges()
    citation_count = citation_graph.number_of_edges()
    shared = path_counts["1"]
    return {
        "papers": len(paper_ids),
        "pairs": len(expected_pairs),
        "logical": logical_count,
        "no_qualifying_relation": len(expected_pairs) - logical_count,
        "citation": citation_count,
        "shared": shared,
        "logical_only": logical_count - shared,
        "citation_only": citation_count - shared,
        "logical_only_across_citation_components": path_counts["no citation path"],
        "logical_only_with_indirect_citation_path": logical_count - shared - path_counts["no citation path"],
        "citation_coverage_of_logical_pairs": shared / logical_count,
        "logical_coverage_of_citation_pairs": shared / citation_count,
        "jaccard_overlap": shared / (logical_count + citation_count - shared),
        "logical_relation_counts": dict(relation_types),
        "cited_logical_relation_counts": dict(cited_relation_types),
        "citation_paths_for_logical_pairs": dict(path_counts),
        "logical_connectivity": graph_summary(logical_graph),
        "citation_connectivity": graph_summary(citation_graph),
    }


COUNT_FIELDS = {
    "pairs": "Evaluated pairs",
    "logical": "Logical pairs",
    "no_qualifying_relation": "No qualifying relation",
    "citation": "Direct citation pairs",
    "shared": "Shared",
    "logical_only": "Logical only",
    "citation_only": "Citation only",
    "logical_only_across_citation_components": "Logical-only across citation components",
    "logical_only_with_indirect_citation_path": "Logical-only with indirect citation path",
}


def verify_counts(computed, documented, group):
    for field, workbook_field in COUNT_FIELDS.items():
        require(computed[field] == documented[workbook_field], f"{group}: workbook disagrees on {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA, help="Directory containing both XLSX files")
    options = parser.parse_args()
    corpus = load_workbook(options.data_dir / CORPUS_FILE, read_only=True, data_only=True)
    edges = load_workbook(options.data_dir / RELATIONS_FILE, read_only=True, data_only=True)
    try:
        papers = records(corpus, "Topic entries")
        publications = records(corpus, "Unique publications")
        citations = records(edges, "Citation edges")
        relations = records(edges, "Logical edges")
        decisions = records(edges, "All pairs")
        comparison = {row["Topic"]: row for row in records(edges, "Summary")}
    finally:
        corpus.close()
        edges.close()
    require(len(papers) == len({row["Topic paper ID"] for row in papers}), "Duplicate topic paper ID")
    require(len(publications) == len({row["Publication ID"] for row in publications})
            and {r["Publication ID"] for r in publications} == {r["Publication ID"] for r in papers},
            "Publication identifiers do not join across corpus worksheets")

    topics = {}
    for topic in sorted({row["Topic code"] for row in papers}):
        inputs = [[row for row in group if row["Topic"] == topic] for group in (citations, relations, decisions)]
        topics[topic] = reconstruct(topic, papers, *inputs)
        verify_counts(topics[topic], comparison[topic], topic)

    totals = {field: sum(topic[field] for topic in topics.values()) for field in COUNT_FIELDS}
    verify_counts(totals, comparison["Total"], "Total")
    for field in ("logical_relation_counts", "cited_logical_relation_counts", "citation_paths_for_logical_pairs"):
        counts = Counter()
        for group in topics.values():
            counts.update(group[field])
        totals[field] = dict(counts)
    print(json.dumps({"topic_entries": len(papers), "distinct_publications": len(publications),
                      "topics": topics, "total": totals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

