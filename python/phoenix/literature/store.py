"""Load, validate, and export the paper corpus.

`research/papers.csv` is a generated export for humans who want a spreadsheet.
Editing it is a no-op that gets overwritten — the YAML files are the source of
truth (TDD §24).
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml
from pydantic import ValidationError

from phoenix.literature.models import Paper


class PaperLoadError(Exception):
    """A paper YAML file failed schema validation. Carries the file path."""

    def __init__(self, path: Path, cause: ValidationError) -> None:
        super().__init__(f"{path}: {cause}")
        self.path = path
        self.cause = cause


def load_paper(path: Path) -> Paper:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return Paper.model_validate(raw)
    except ValidationError as e:
        raise PaperLoadError(path, e) from e


def load_all(papers_dir: Path) -> list[Paper]:
    """Load every paper. Fails loudly and names every bad file at once.

    A partial corpus silently missing one broken entry is worse than a load that
    refuses until every record is honest — this is a small, human-curated corpus,
    not a stream where partial availability matters.
    """
    papers: list[Paper] = []
    errors: list[PaperLoadError] = []
    for path in sorted(papers_dir.glob("*.yaml")):
        try:
            papers.append(load_paper(path))
        except PaperLoadError as e:
            errors.append(e)
    if errors:
        detail = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(f"{len(errors)} paper record(s) failed validation:\n{detail}")
    ids = [p.paper_id for p in papers]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RuntimeError(f"duplicate paper_id(s) across files: {sorted(dupes)}")
    return papers


CSV_COLUMNS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "venue",
    "categories",
    "hardware",
    "methodology_type",
    "measured_on_physical_hardware",
    "n_claims",
    "read_status",
    "phoenix_relevance",
    "doi",
    "arxiv",
    "url",
]


def to_csv_row(p: Paper) -> dict[str, str]:
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "authors": "; ".join(p.authors),
        "year": str(p.year) if p.year is not None else "",
        "venue": p.venue.name,
        "categories": "; ".join(p.categories),
        "hardware": "; ".join(h.kind for h in p.hardware_studied),
        "methodology_type": p.methodology.type.value,
        "measured_on_physical_hardware": str(p.methodology.measured_on_physical_hardware),
        "n_claims": str(len(p.claims)),
        "read_status": p.provenance.read_status.value,
        "phoenix_relevance": p.phoenix_relevance.relevance.value,
        "doi": p.identifiers.doi or "",
        "arxiv": p.identifiers.arxiv or "",
        "url": p.identifiers.url or "",
    }


def export_csv(papers: list[Paper], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for p in sorted(papers, key=lambda p: p.paper_id):
            writer.writerow(to_csv_row(p))
    return out_path


def corpus_summary(papers: list[Paper]) -> dict[str, int]:
    """Counts that matter for the DoD's ≥50-paper / FULL_READ-or-SKIMMED target."""
    from phoenix.literature.models import ReadStatus

    citable = sum(1 for p in papers if p.provenance.read_status != ReadStatus.ABSTRACT_ONLY)
    return {
        "total": len(papers),
        "full_read": sum(1 for p in papers if p.provenance.read_status == ReadStatus.FULL_READ),
        "skimmed": sum(1 for p in papers if p.provenance.read_status == ReadStatus.SKIMMED),
        "abstract_only": sum(
            1 for p in papers if p.provenance.read_status == ReadStatus.ABSTRACT_ONLY
        ),
        "citable": citable,
        "total_claims": sum(len(p.claims) for p in papers),
    }
