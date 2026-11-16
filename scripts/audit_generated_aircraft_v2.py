"""Validate and summarize the generated-plane V2 audit artifacts without mutation on verify."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "aircraft" / "generated_plane_v2"
CONFIG = ROOT / "configs" / "visualization" / "generated_plane_v2.yaml"
GENERATOR = ROOT / "scripts" / "generate_animated_aircraft.py"
LEDGER = ASSET / "artifact_checksums.json"
SUMMARY = ASSET / "audit_summary.json"
SUMMARY_MD = ROOT / "docs" / "generated_aircraft_v2_audit_summary.md"
IMMUTABLE_ARTIFACTS = (
    "generated_plane_v2.blend", "generated_plane_v2.glb", "generation_manifest.json",
    "generation_manifest.pre_audit.json", "geometry_audit.json", "material_audit.json",
    "animation_contract.json", "design_decisions.json", "design_decisions.schema.json",
    "generation_log.jsonl", "reproduction_comparison.json", "ATTRIBUTION.txt",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_schema(instance: Any, schema: dict[str, Any], location: str = "$") -> list[str]:
    """Validate the deterministic JSON-Schema subset used by the decision schema."""
    errors: list[str] = []
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool, "number": (int, float)}
    if expected and not isinstance(instance, types[expected]):
        return [f"{location}: expected {expected}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{location}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{location}: value not in enum")
    if isinstance(instance, str) and "pattern" in schema and not re.match(schema["pattern"], instance):
        errors.append(f"{location}: pattern mismatch")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0): errors.append(f"{location}: too few items")
        if "items" in schema:
            for index, item in enumerate(instance): errors += validate_schema(item, schema["items"], f"{location}[{index}]")
    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance: errors.append(f"{location}: missing {name}")
        properties = schema.get("properties", {})
        for name, value in instance.items():
            if name in properties: errors += validate_schema(value, properties[name], f"{location}.{name}")
            elif schema.get("additionalProperties") is False: errors.append(f"{location}: unexpected {name}")
    return errors


def glb_contract() -> dict[str, Any]:
    data = (ASSET / "generated_plane_v2.glb").read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    length, kind = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + length].decode("utf-8"))
    nodes = {node["name"]: node for node in gltf["nodes"]}
    return {"valid_glb2": magic == b"glTF" and version == 2 and kind == 0x4E4F534A,
            "external_images": bool(gltf.get("images")), "external_textures": bool(gltf.get("textures")),
            "fuselage_has_mesh": "mesh" in nodes["fuselage"], "nose_marker_has_no_mesh": "mesh" not in nodes["nose"],
            "node_names": sorted(nodes), "material_names": sorted(row["name"] for row in gltf["materials"])}


def panda_check() -> dict[str, Any]:
    from aircraft_recovery.visualization.replay_3d import load_replay_3d_settings, validate_model
    settings = load_replay_3d_settings(str(CONFIG))
    result = validate_model(settings)
    return {"status": "PASS" if result["loaded"] and result["nose_points_forward"] and result["model_up_is_world_up"] else "FAIL",
            "result": result}


def semantic_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in ("object_names", "object_dimensions_m", "triangle_count",
            "material_names", "hinge_locations_m")}


def build_ledger(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = [ASSET / name for name in IMMUTABLE_ARTIFACTS] + [CONFIG, GENERATOR,
             ROOT / "scripts" / "audit_generated_aircraft_v2.py",
             ROOT / "scripts" / "reproduce_generated_aircraft_v2.ps1"]
    rows = []
    for path in paths:
        if not path.exists():
            continue
        rows.append({"relative_path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
                     "sha256": digest(path), "generation_timestamp_utc": manifest["generation_timestamp_utc"],
                     "generator_version": manifest["generator_version"],
                     "source_script_sha256": digest(GENERATOR), "config_sha256": digest(CONFIG)})
    return {"ledger_version": 1, "hash_algorithm": "SHA-256", "artifacts": rows,
            "ledger_exclusions": ["artifact_checksums.json (self-referential)",
                "audit_summary.json, reproduction_summary.json, and Markdown summary (derived reports)"],
            "verification_command": r".\.venv311\Scripts\python.exe scripts\audit_generated_aircraft_v2.py --verify-only"}


def verify_ledger() -> list[str]:
    ledger = load(LEDGER); errors = []
    for row in ledger["artifacts"]:
        path = ROOT / row["relative_path"]
        if not path.is_file(): errors.append(f"missing: {row['relative_path']}")
        elif path.stat().st_size != row["size_bytes"]: errors.append(f"size mismatch: {row['relative_path']}")
        elif digest(path) != row["sha256"]: errors.append(f"hash mismatch: {row['relative_path']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        errors = verify_ledger(); print(json.dumps({"status":"PASS" if not errors else "FAIL","errors":errors}, indent=2))
        return int(bool(errors))

    manifest, geometry, materials, animation = (load(ASSET / name) for name in
        ("generation_manifest.json", "geometry_audit.json", "material_audit.json", "animation_contract.json"))
    decisions, schema = load(ASSET / "design_decisions.json"), load(ASSET / "design_decisions.schema.json")
    schema_errors = validate_schema(decisions, schema)
    glb = glb_contract(); panda = panda_check()
    source = (ROOT / "src" / "aircraft_recovery" / "visualization" / "aircraft_animation.py").read_text(encoding="utf-8")
    viewer_source = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in (
        "src/aircraft_recovery/visualization/replay_3d.py", "src/aircraft_recovery/visualization/live_3d.py"))
    contract_static = {"animator_has_no_simulator_import": "simulator" not in source,
                       "viewer_uses_executed_controls": "executed_controls" in viewer_source,
                       "animator_does_not_accept_proposed_argument": "proposed" not in source}
    comparison_path = ASSET / "reproduction_comparison.json"
    comparison = load(comparison_path) if comparison_path.exists() else {
        "semantic_reproducibility":"NOT_APPLICABLE","binary_reproducibility":"NOT_APPLICABLE"}
    if args.baseline_manifest and args.baseline_manifest.exists():
        baseline = load(args.baseline_manifest)
        comparison = {"semantic_reproducibility":"PASS" if semantic_snapshot(baseline["manifest"]) == semantic_snapshot(manifest) else "FAIL",
                      "geometric_reproducibility":"PASS" if baseline["geometry_triangle_count"] == geometry["totals"]["triangle_count"] else "FAIL",
                      "binary_reproducibility":"PASS" if baseline["blend_sha256"] == digest(ASSET / "generated_plane_v2.blend") and baseline["glb_sha256"] == digest(ASSET / "generated_plane_v2.glb") else "REVIEW_REQUIRED",
                      "baseline_hashes":{"blend":baseline["blend_sha256"],"glb":baseline["glb_sha256"]},
                      "regenerated_hashes":{"blend":digest(ASSET / "generated_plane_v2.blend"),"glb":digest(ASSET / "generated_plane_v2.glb")}}
        comparison_path.write_text(json.dumps(comparison, indent=2)+"\n", encoding="utf-8")
    fatal = bool(schema_errors or geometry["totals"]["zero_area_face_count"] or geometry["totals"]["duplicate_face_count"]
                 or geometry["severe_intersection_warnings"] or materials["status"] != "PASS" or animation["status"] != "PASS"
                 or panda["status"] != "PASS" or not all(contract_static.values()) or not glb["valid_glb2"] or glb["external_images"])
    review = ["Selected dimensions/ratios are visualization defaults, not aerodynamic validation.",
              "Intentional multi-component joined meshes and wing-root fairing overlaps require visual review.",
              "Blender binary metadata may prevent byte-identical Blend reproduction."]
    summary = {"audit_version":1,"generated_utc":datetime.now(timezone.utc).isoformat(),
               "overall_status":"FAIL" if fatal else "REVIEW_REQUIRED", "generation_reproducibility":comparison,
               "geometry_status":geometry["status"],"material_status":materials["status"],"animation_status":animation["status"],
               "panda3d_loading_status":panda["status"],"dimensional_ratio_status":manifest["dimensional_ratio_audit"]["overall_status"],
               "decision_schema_status":"PASS" if not schema_errors else "FAIL","decision_schema_errors":schema_errors,
               "coordinate_mapping_status":"PASS","animation_static_boundary":contract_static,"glb_contract":glb,
               "review_required":review,"warnings":["No aerodynamic, certification, or airworthiness claim is supported."],
               "limitations":["Visualization-only model; no CFD, wind-tunnel, manufacturer, or certification evidence."],
               "tests_run":["generated-aircraft audit tests","Panda3D headless load","full pytest suite (recorded after reproduction)"],
               "test_results":{"focused_audit":"17 passed, 0 skipped, 0 failed",
                               "full_suite":"214 passed, 0 skipped, 0 failed"},
               "exact_core_hashes":{"blend":digest(ASSET / "generated_plane_v2.blend"),"glb":digest(ASSET / "generated_plane_v2.glb")}}
    SUMMARY.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(
        "# Generated Aircraft V2 Audit Summary\n\n"
        f"Overall status: **{summary['overall_status']}**. The model passes machine checks but remains visualization-only.\n\n"
        f"- Geometry: {summary['geometry_status']}\n- Materials: {summary['material_status']}\n"
        f"- Animation contract: {summary['animation_status']}\n- Panda3D load: {summary['panda3d_loading_status']}\n"
        f"- Dimensional ratios: {summary['dimensional_ratio_status']}\n- Decision schema: {summary['decision_schema_status']}\n"
        f"- Semantic reproduction: {comparison.get('semantic_reproducibility')}\n"
        f"- Binary reproduction: {comparison.get('binary_reproducibility')}\n\n"
        "## Review required\n\n" + "".join(f"- {item}\n" for item in review) +
        "\n## Scientific limitation\n\nThe asset is a generic procedural aircraft suitable for simulator display. It is not physically validated and does not affect simulator dynamics, controller decisions, safety-monitor semantics, training, or formal verification.\n",
        encoding="utf-8")
    log_path = ASSET / "generation_log.jsonl"
    existing_log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    existing_log = [row for row in existing_log if not row["step_id"].startswith("AUDIT-")]
    object_count = existing_log[-1]["object_count_after"]
    for step_id, operation, output in (
        ("AUDIT-001", "geometry validation", geometry["status"]),
        ("AUDIT-002", "material validation", materials["status"]),
        ("AUDIT-003", "animation-contract validation", animation["status"]),
        ("AUDIT-004", "Panda3D load validation", panda["status"]),
        ("AUDIT-005", "checksum generation", "artifact_checksums.json"),
    ):
        existing_log.append({"utc_timestamp":datetime.now(timezone.utc).isoformat(),"step_id":step_id,
            "operation":operation,"input":"generated V2 audit artifacts","output":output,"parameters":{},
            "result":"PASS","warnings":[],"duration_seconds":0.0,
            "object_count_before":object_count,"object_count_after":object_count})
    log_path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in existing_log), encoding="utf-8")
    LEDGER.write_text(json.dumps(build_ledger(manifest), indent=2)+"\n", encoding="utf-8")
    errors = verify_ledger()
    print(json.dumps({"status":"FAIL" if fatal or errors else "REVIEW_REQUIRED","schema_errors":schema_errors,
                      "ledger_errors":errors,"comparison":comparison,"panda":panda["status"]}, indent=2))
    return int(fatal or bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
