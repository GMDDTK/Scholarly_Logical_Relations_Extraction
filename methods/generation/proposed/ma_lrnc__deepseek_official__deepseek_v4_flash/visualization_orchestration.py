from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .interactive_visualization import render_interactive_html
from .network_visualization import write_network_data
from .publication_visualization import render_publication_figures
from .result_paths import TopicResultPaths
from .topic_input import TopicInputSet


@dataclass(frozen=True)
class TopicVisualizationBuildResult:
    input_fingerprint: str
    output_paths: tuple[Path, ...]
    reused: bool


def build_topic_visualization(
    *,
    topic_input: TopicInputSet,
    paths: TopicResultPaths,
) -> TopicVisualizationBuildResult:
    network_data = write_network_data(topic_input=topic_input, paths=paths)
    publication = render_publication_figures(
        bundle=network_data.bundle,
        output_root=paths.network_root / "figures",
        temp_root=paths.temp_root / "network" / "figures",
    )
    interactive = render_interactive_html(
        bundle=network_data.bundle,
        output_path=paths.network_root / "index.html",
        temp_root=paths.temp_root / "network" / "html",
    )
    output_paths = (
        network_data.output_path,
        interactive.output_path,
        *publication.output_paths,
    )
    if not all(path.is_file() for path in output_paths):
        missing = [str(path) for path in output_paths if not path.is_file()]
        raise RuntimeError(f"visualization output set is incomplete: {missing}")
    return TopicVisualizationBuildResult(
        input_fingerprint=network_data.bundle.input_fingerprint,
        output_paths=output_paths,
        reused=(
            network_data.reused
            and publication.reused
            and interactive.reused
        ),
    )
