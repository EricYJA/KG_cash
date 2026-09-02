"""Shared ablation reporting for the parametric (no-KG) experiment runners.

scripts/run_rog_parametric_experiment.py and run_tog_parametric_experiment.py
both finish the same way: one row for the no-KG condition they just ran, plus one
row per KG-grounded condition read off the matching cache experiment's outputs.
That side-by-side IS the ablation -- the two conditions differ in exactly one
thing, whether anything from the knowledge graph reaches the prompt, so the gap
between them is what the KG contributes over what the model already remembers.

Kept apart from _rog_common / _tog_common because it is the one piece both
runners share: the systems differ (RoG scores 0-100, ToG 0-1; RoG has a planner
stage, ToG a traversal loop), but the report shape does not.

Standard library only, like the runners themselves: these orchestrate a container
(RoG) or a conda env (ToG) and must start under whatever interpreter is at hand.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# A column is (row key, printed header, format spec). The spec is applied only
# to numbers; anything missing or non-numeric prints as "-" so a partially
# populated comparison row still lines up instead of crashing the report.
Column = tuple[str, str, str]

# Width of the leading condition column. Policy names are short; the delta row's
# label is the longest thing that lands here, so it sets the floor.
_LABEL_W = 30


def _label(text: str) -> str:
    """Left-align a row label in the condition column, clipping an overlong one.

    Clipped rather than allowed to push the row wider: a single long tag would
    otherwise shift that row's numbers out from under their headers.
    """
    if len(text) > _LABEL_W - 1:
        text = text[: _LABEL_W - 2] + "~"
    return format(text, f"<{_LABEL_W}")


def _head_spec(spec: str) -> str:
    """Text spec matching a value spec: same alignment and width, no precision."""
    body = spec.lstrip("<>^=")
    align = spec[: len(spec) - len(body)] or ">"
    return f"{align}{body.split('.')[0] or ''}"


def fmt_cell(value: object, spec: str) -> str:
    """Format one table cell, or a right-aligned dash when there is no number.

    Booleans are excluded deliberately: `True` is an int in Python and would
    print as 1, quietly turning a flag into a score. Counts read back out of a
    CSV arrive as floats, so an integer column narrows them rather than printing
    a question count as "50.0".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return format("-", _head_spec(spec))
    if "." not in spec and isinstance(value, float) and value.is_integer():
        value = int(value)
    return format(value, spec)


def find_row(rows: list[dict], condition: str) -> dict | None:
    for row in rows:
        if row.get("condition") == condition:
            return row
    return None


# Vendor presets, mirroring src/ToG-cache/ToG/llm_config.py. Duplicated rather
# than imported: these runners start under whatever interpreter is at hand and
# must not depend on src/ being importable.
VENDOR_DEFAULT_MODEL = {
    "tamu": "protected.Claude-Haiku-4.5",
    "openai": "gpt-4.1-mini",
    "google": "gemini-3-flash-preview",
}


def resolve_model(vendor: str, model: str | None) -> str:
    """The model a run will ACTUALLY use, with the vendor default filled in.

    An unset --model is not "no model", it is the vendor preset. Recording the
    raw empty string is how four ablations came to be named `gemini_*` while
    every one of them ran on the tamu default -- the run metadata said
    `"model": ""` and nothing contradicted the directory name.
    """
    return (model or "").strip() or VENDOR_DEFAULT_MODEL.get(vendor, "unknown")


def guard_run_identity(out_dir: Path, identity: dict, *, fresh: bool = False) -> None:
    """Pin a run's identity on first use; refuse to resume a tag under a new one.

    The parametric runners resume by default, and their output files are named
    for the condition alone -- nothing in `parametric_no_kg.jsonl` records which
    model produced it. Re-using a tag after switching models therefore appends
    one model's answers to another's and scores the mixture. This is the same
    protection compare_cache_accuracy.py:60 gives the cache experiment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "run_config.json"
    if fresh or not manifest.exists():
        manifest.write_text(json.dumps({"identity": identity}, indent=2))
        return
    try:
        previous = json.loads(manifest.read_text()).get("identity", {})
    except json.JSONDecodeError:
        previous = {}
    differing = [k for k in identity if previous.get(k) != identity[k]]
    if not differing:
        return
    detail = "\n".join(f"    {k}: {previous.get(k)!r} -> {identity[k]!r}"
                       for k in differing)
    raise SystemExit(
        f"{out_dir} already holds a different run's results:\n{detail}\n"
        f"Resuming would append these answers to those files and score the "
        f"mixture.\nPass --fresh to redo this tag, or use a new --run-tag."
    )


def warn_tag_model_mismatch(run_tag: str, model: str) -> None:
    """Warn when a run tag advertises a model the run is not actually using.

    Cheap, and it catches the exact mistake that cost four ablations: a tag
    named for Gemini on a run that resolved to the tamu Haiku default.
    """
    tag = (run_tag or "").lower()
    for hint in ("gemini", "gpt", "claude", "haiku", "sonnet", "opus"):
        if hint in tag and hint not in model.lower():
            print(f"[WARN] run tag {run_tag!r} mentions {hint!r} but this run "
                  f"resolves to model {model!r}. Pass --model, or --env-file "
                  f"pointing at an env file that sets MODEL.")
            return


def emit_ablation(
    *,
    out_dir: Path,
    rows: list[dict],
    columns: list[Column],
    subject: str,
    reference: str | None,
    title: str,
    count_key: str | None = None,
    notes: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Print the ablation table and write ablation.json / ablation.csv.

    `subject` is the no-KG condition this run produced; `reference` is the
    KG-grounded condition it is being ablated against (normally the uncached
    baseline, so the comparison isolates the KG rather than a cache policy).
    A missing reference is reported, not faked: the subject row still prints and
    is still written, with `delta` left null, because a run whose comparison
    partner has not happened yet is incomplete, not wrong.

    `count_key` names the column holding each row's question count. When the two
    rows disagree on it the delta is still shown but called out as covering
    different question sets -- the easy way to get a meaningless ablation is to
    run the no-KG side with a different --limit than the KG side.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_row = find_row(rows, subject)
    reference_row = find_row(rows, reference) if reference else None

    header = f"{'condition':<{_LABEL_W}}" + "".join(
        format(head, _head_spec(spec)) for key, head, spec in columns if key != "condition"
    )
    print()
    print("=" * max(len(header), 64))
    print(f">>> {title}")
    print("=" * max(len(header), 64))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{_label(str(row.get('condition', '?')))}" + "".join(
            fmt_cell(row.get(key), spec) for key, _, spec in columns if key != "condition"
        )
        print(line)
    print("-" * len(header))

    delta: dict[str, float] | None = None
    if subject_row is None:
        print("[warn] the parametric row is missing; nothing to ablate")
    elif reference_row is None:
        print(f"[warn] no KG-grounded row {reference!r} to compare against -- run the "
              f"matching cache experiment, or point --compare-dir at its results")
    else:
        delta = {}
        for key, _, spec in columns:
            if key == "condition":
                continue
            a, b = subject_row.get(key), reference_row.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                    and not isinstance(a, bool) and not isinstance(b, bool):
                delta[key] = a - b
        line = f"{_label(f'delta vs {reference}')}" + "".join(
            fmt_cell(delta.get(key), spec) for key, _, spec in columns if key != "condition"
        )
        print(line)
        print(f"\n>>> a negative delta is the knowledge graph earning its place: that "
              f"much of {reference!r}'s score the model could NOT supply from its own "
              f"weights.")
        if count_key and delta.get(count_key):
            print(f"[warn] the two conditions cover different question counts "
                  f"({subject_row.get(count_key)} vs {reference_row.get(count_key)}) -- "
                  f"re-run them with the same --limit before reading the delta as a "
                  f"result")
    for note in notes:
        print(f">>> {note}")

    payload = {
        "title": title,
        "subject": subject,
        "reference": reference,
        "delta": delta,
        "rows": rows,
    }
    json_path = out_dir / "ablation.json"
    csv_path = out_dir / "ablation.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    fieldnames = ["condition"] + [key for key, _, _ in columns if key != "condition"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {json_path} and {csv_path}")
    return json_path, csv_path
