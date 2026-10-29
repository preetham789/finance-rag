from pathlib import Path
from src.evaluation.dataset import (
    EVAL_DATASET,
    get_eval_dataset,
    to_ragas_format,
    export_to_jsonl,
    load_from_jsonl,
)
from src.evaluation.ragas_eval import TEST_DATASET


def test_eval_dataset_size_and_categories():
    assert len(EVAL_DATASET) >= 50
    categories = set(d.get("category") for d in EVAL_DATASET)
    assert "single_fact" in categories
    assert "numeric" in categories
    assert "out_of_corpus" in categories
    assert "temporal_trap" in categories
    assert "adversarial" in categories
    assert "multi_company" in categories


def test_get_eval_dataset_filters():
    rbi_items = get_eval_dataset(company="RBI")
    assert len(rbi_items) >= 5
    for item in rbi_items:
        assert item.get("company") == "RBI"

    oos_items = get_eval_dataset(category="out_of_corpus")
    assert len(oos_items) >= 5
    for item in oos_items:
        assert item.get("expect", {}).get("refuse") is True


def test_to_ragas_format():
    ragas_items = to_ragas_format(EVAL_DATASET)
    assert len(ragas_items) > 0
    # Refusal questions should not be in RAGAS format
    for item in ragas_items:
        assert "question" in item
        assert "ground_truth" in item
        assert item["ground_truth"] != ""


def test_test_dataset_synced():
    assert len(TEST_DATASET) >= 30
    assert TEST_DATASET[0]["question"] == EVAL_DATASET[0]["question"]


def test_export_and_load_jsonl(tmp_path: Path):
    out_file = tmp_path / "test_eval.jsonl"
    count = export_to_jsonl(out_file)
    assert count == len(EVAL_DATASET)
    assert out_file.exists()

    loaded = load_from_jsonl(out_file)
    assert len(loaded) == count
    assert loaded[0]["id"] == EVAL_DATASET[0]["id"]
    assert loaded[0]["question"] == EVAL_DATASET[0]["question"]
