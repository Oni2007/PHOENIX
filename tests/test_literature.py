"""Literature database schema and store (TDD §24)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phoenix.literature.models import Paper
from phoenix.literature.store import corpus_summary, export_csv, load_all, load_paper

PAPERS_DIR = Path("research/papers")

MINIMAL: dict = {
    "schema_version": "1.0.0",
    "paper_id": "PAP-9001",
    "title": "Minimal Test Paper",
    "authors": ["A. Author"],
    "venue": {"name": "Test Venue", "type": "CONFERENCE"},
    "identifiers": {"doi": "10.0000/test"},
    "methodology": {"type": "EXPERIMENTAL", "measured_on_physical_hardware": True},
    "phoenix_relevance": {"relevance": "LOW"},
    "provenance": {"added_by": "test", "added_on": "2026-08-20", "read_status": "SKIMMED"},
}


def test_minimal_record_validates() -> None:
    p = Paper.model_validate(MINIMAL)
    assert p.paper_id == "PAP-9001"


def test_abstract_only_cannot_carry_claims() -> None:
    """TDD §24: 'A paper may never be cited in PHOENIX output at ABSTRACT_ONLY status.'"""
    bad = dict(MINIMAL)
    bad["provenance"] = {**MINIMAL["provenance"], "read_status": "ABSTRACT_ONLY"}
    bad["claims"] = [
        {
            "claim_id": "PAP-9001-C1",
            "statement": "x",
            "evidence_class": "MEASURED",
            "scope": "s",
            "confidence_in_claim": "LOW",
        }
    ]
    with pytest.raises(ValidationError, match="ABSTRACT_ONLY"):
        Paper.model_validate(bad)


def test_abstract_only_with_no_claims_is_fine() -> None:
    ok = dict(MINIMAL)
    ok["provenance"] = {**MINIMAL["provenance"], "read_status": "ABSTRACT_ONLY"}
    Paper.model_validate(ok)  # must not raise


def test_claim_id_must_be_namespaced_under_paper_id() -> None:
    bad = dict(MINIMAL)
    bad["claims"] = [
        {
            "claim_id": "WRONG-PREFIX-C1",
            "statement": "x",
            "evidence_class": "MEASURED",
            "scope": "s",
            "confidence_in_claim": "LOW",
        }
    ]
    with pytest.raises(ValidationError, match="namespaced"):
        Paper.model_validate(bad)


def test_duplicate_claim_ids_within_a_paper_are_rejected() -> None:
    bad = dict(MINIMAL)
    claim = {
        "claim_id": "PAP-9001-C1",
        "statement": "x",
        "evidence_class": "MEASURED",
        "scope": "s",
        "confidence_in_claim": "LOW",
    }
    bad["claims"] = [claim, dict(claim)]
    with pytest.raises(ValidationError, match="duplicate"):
        Paper.model_validate(bad)


def test_numeric_claim_quantity_requires_a_unit() -> None:
    bad = dict(MINIMAL)
    bad["claims"] = [
        {
            "claim_id": "PAP-9001-C1",
            "statement": "x",
            "quantity": {"name": "throughput", "value": 92.0},  # no unit
            "evidence_class": "MEASURED",
            "scope": "s",
            "confidence_in_claim": "LOW",
        }
    ]
    with pytest.raises(ValidationError, match="unit"):
        Paper.model_validate(bad)


def test_identifiers_require_at_least_one_of_doi_arxiv_url() -> None:
    bad = dict(MINIMAL)
    bad["identifiers"] = {}
    with pytest.raises(ValidationError):
        Paper.model_validate(bad)


def test_unknown_field_is_rejected_not_ignored() -> None:
    bad = dict(MINIMAL)
    bad["definitely_a_typo"] = True
    with pytest.raises(ValidationError):
        Paper.model_validate(bad)


# ── against the real seeded corpus ─────────────────────────────────────────────


def test_seeded_corpus_loads_and_validates() -> None:
    corpus = load_all(PAPERS_DIR)
    assert len(corpus) >= 6


def test_seeded_corpus_has_no_duplicate_paper_ids() -> None:
    corpus = load_all(PAPERS_DIR)
    ids = [p.paper_id for p in corpus]
    assert len(ids) == len(set(ids))


def test_seeded_corpus_every_claim_has_scope_and_excludes_considered() -> None:
    """The load-bearing fields per TDD §24: scope is mandatory; excludes may be
    empty but must be a real list, not omitted silently."""
    corpus = load_all(PAPERS_DIR)
    for paper in corpus:
        for claim in paper.claims:
            assert claim.scope.strip(), f"{claim.claim_id} has an empty scope"
            assert isinstance(claim.excludes, list)


def test_seeded_corpus_abstract_only_papers_carry_no_claims() -> None:
    corpus = load_all(PAPERS_DIR)
    for paper in corpus:
        if paper.provenance.read_status.value == "ABSTRACT_ONLY":
            assert paper.claims == [], f"{paper.paper_id} is ABSTRACT_ONLY but has claims"


def test_csv_export_round_trips_row_count(tmp_path: Path) -> None:
    corpus = load_all(PAPERS_DIR)
    out = export_csv(corpus, tmp_path / "papers.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(corpus) + 1  # header + one row per paper


def test_corpus_summary_counts_are_internally_consistent() -> None:
    corpus = load_all(PAPERS_DIR)
    summary = corpus_summary(corpus)
    assert summary["total"] == len(corpus)
    assert summary["full_read"] + summary["skimmed"] + summary["abstract_only"] == summary["total"]
    assert summary["citable"] == summary["full_read"] + summary["skimmed"]


def test_load_paper_reports_the_offending_file_on_failure(tmp_path: Path) -> None:
    bad_file = tmp_path / "PAP-0000.yaml"
    bad_file.write_text("paper_id: not even close to valid\n")
    with pytest.raises(Exception) as exc_info:
        load_paper(bad_file)
    assert str(bad_file) in str(exc_info.value) or "PAP-0000" in str(bad_file)


def test_duplicate_paper_id_across_two_files_is_rejected(tmp_path: Path) -> None:
    import yaml

    record = dict(MINIMAL)
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(record))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(record))
    with pytest.raises(RuntimeError, match="duplicate"):
        load_all(tmp_path)
