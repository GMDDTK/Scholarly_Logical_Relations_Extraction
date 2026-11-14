import hashlib
import json
from pathlib import Path

from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.interactive_visualization import (
    render_interactive_html,
)
from methods.generation.proposed.ma_lrnc__deepseek_official__deepseek_v4_flash.network_visualization import (
    build_network_bundle,
)

from test_network_data import _write_complete_inputs


def test_vendored_echarts_asset_matches_manifest_and_license() -> None:
    asset_root = (
        Path(__file__).resolve().parents[4]
        / "methods"
        / "generation"
        / "proposed"
        / "ma_lrnc__deepseek_official__deepseek_v4_flash"
        / "visualization_assets"
    )
    manifest = json.loads(
        (asset_root / "manifest.json").read_text(encoding="utf-8")
    )
    runtime = (asset_root / manifest["runtime_file"]).read_bytes()
    license_text = (asset_root / manifest["license_file"]).read_text(
        encoding="utf-8"
    )

    assert manifest["version"] == "6.1.0"
    assert hashlib.sha256(runtime).hexdigest() == manifest["runtime_sha256"]
    assert "Apache License" in license_text


def test_interactive_html_is_self_contained_svg_and_traceable(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)

    result = render_interactive_html(
        bundle=bundle,
        output_path=paths.network_root / "index.html",
        temp_root=paths.temp_root / "network",
    )

    html = result.output_path.read_text(encoding="utf-8")
    assert result.reused is False
    assert "<!doctype html>" in html.lower()
    assert '<script src=' not in html.lower()
    assert 'src="http' not in html.lower()
    assert 'href="http' not in html.lower()
    assert 'fetch("http' not in html.lower()
    assert "//cdn" not in html.lower()
    assert bundle.input_fingerprint in html
    assert 'renderer: "svg"' in html
    assert 'id="projection-select"' in html
    assert 'id="relation-filters"' in html
    assert 'id="network-chart"' in html
    assert 'id="matrix-chart"' in html
    assert 'id="detail-panel"' in html
    assert "manual_review_required" in html
    assert "待人工复核结论" in html
    assert (
        'DATA.pairs.filter(o => o.review_status === "manual_review_required").length'
        in html
    )
    assert "Apache License" in html
    assert result.output_path.stat().st_size > 100_000


def test_interactive_html_escapes_embedded_model_text(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    profile_path = paths.final_profile_root / "P001.json"
    malicious = profile_path.read_text(encoding="utf-8").replace(
        "P001研究问题",
        '</script><script>window.injected=true</script>',
    )
    profile_path.write_text(malicious, encoding="utf-8")
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)

    result = render_interactive_html(
        bundle=bundle,
        output_path=paths.network_root / "index.html",
        temp_root=paths.temp_root / "network",
    )

    html = result.output_path.read_text(encoding="utf-8")
    assert '</script><script>window.injected=true</script>' not in html
    assert r"\u003c/script\u003e" in html
    assert "textContent" in html


def test_interactive_html_reuses_matching_output_and_rebuilds_damage(
    tmp_path: Path,
) -> None:
    topic_input, paths = _write_complete_inputs(tmp_path)
    bundle = build_network_bundle(topic_input=topic_input, paths=paths)
    output_path = paths.network_root / "index.html"
    temp_root = paths.temp_root / "network"

    first = render_interactive_html(
        bundle=bundle,
        output_path=output_path,
        temp_root=temp_root,
    )
    original_mtime = output_path.stat().st_mtime_ns
    second = render_interactive_html(
        bundle=bundle,
        output_path=output_path,
        temp_root=temp_root,
    )
    assert first.reused is False
    assert second.reused is True
    assert output_path.stat().st_mtime_ns == original_mtime

    output_path.write_text("damaged", encoding="utf-8")
    rebuilt = render_interactive_html(
        bundle=bundle,
        output_path=output_path,
        temp_root=temp_root,
    )
    assert rebuilt.reused is False
    assert bundle.input_fingerprint in output_path.read_text(encoding="utf-8")
    assert not list(temp_root.glob("*.tmp"))
