import json

import artifacts as artifacts
import manifest as config
from artifacts import write_json
from execution import experiment_status as status
from grounding.element_features import FEATURE_NAMES as FEATURE_NAMES
from grounding.element_features import features as features
from grounding.element_features import (
    first_window_features,
    restrict_candidates,
)
from grounding.evidence import log_sigmoid, row_weights
from grounding.evidence import proposal_info
from manifest import load_recipe, resolve

RECIPE = load_recipe("element")
TRAIN = resolve(RECIPE["inputs"]["evidence"])
FEATURES = resolve(RECIPE["inputs"]["features"])
PREDICTIONS = resolve(RECIPE["inputs"]["predictions"])
HEAD = resolve(RECIPE["inference"]["head"])
SEEDS = tuple(RECIPE["seeds"])


def inputs(split, reference):
    if split == "valid":
        labels = artifacts.read_json(TRAIN / "supervision.json")
        rows = [
            dict(r, labels=labels[r["action_key"]])
            for r in artifacts.read_json(TRAIN / "inference_inputs.json")
            if r["partition"] == "valid"
        ]
        tasks = {r["task_id"] for r in rows}
        baseline = [
            r
            for r in artifacts.read_json(TRAIN / "fit_valid_baseline.json")
            if r["annotation_id"] in tasks
        ]
        features = [
            r
            for r in artifacts.read_json(FEATURES / "train.json")
            if r["partition"] == "valid"
        ]
        records = artifacts.read_json(PREDICTIONS / "main-valid.json")
        return rows, baseline, features, records, TRAIN / "points", TRAIN / "semantics"
    return (
        artifacts.read_json(config.frozen_input("anchored_rows", split, reference)),
        artifacts.read_json(
            config.frozen_input("grounding_baseline", split, reference)
        ),
        artifacts.read_json(FEATURES / f"{split}.json"),
        artifacts.read_json(PREDICTIONS / f"main-{split}.json"),
        config.frozen_input("grounding_points", split, reference),
        config.frozen_input("grounding_semantics", split, reference),
    )


def evidence_rows(rows, point_dir, sem_dir, data):
    """Compact variable-length log-P/prior arrays for the fixed-score sweep."""
    result, smallest, supports = [], {}, {}
    for row, f in zip(rows, data):
        assert row["action_key"] == f["action_key"]
        points = artifacts.read_json(point_dir / f"{row['index']}.json")
        sem = artifacts.read_json(sem_dir / f"{row['index']}.json")
        logits = sem["semantic_logits"]
        hits, native, _ = proposal_info(row, points)
        assert sorted(hits) == f["candidate_ids"]
        priors = row_weights(row)
        result.append(
            dict(
                key=row["action_key"],
                ids=f["candidate_ids"],
                fallback=f["fallback"],
                logps=[
                    [h["prediction"]["log_probability"] for h in hits[i]]
                    for i in f["candidate_ids"]
                ],
                priors=[
                    [priors[h["window"]] for h in hits[i]] for i in f["candidate_ids"]
                ],
                semantics=[log_sigmoid(logits[i]) for i in f["candidate_ids"]],
            )
        )
        smallest[row["action_key"]] = native
        supports[row["action_key"]] = {
            i: {h["window"] for h in hs} for i, hs in hits.items()
        }
    return result, smallest, supports




def load_evidence(rows, point_dir, semantic_dir):
    result = []
    for row in rows:
        points = artifacts.read_json(point_dir / f"{row['index']}.json")
        semantic = artifacts.read_json(semantic_dir / f"{row['index']}.json")
        assert points["action_key"] == semantic["action_key"] == row["action_key"]
        logits = semantic["semantic_logits"]
        result.append(features(row, points, logits))
    return result


def prepare(output, source):
    rows = artifacts.read_json(source / "inference_inputs.json")
    reference = config.load()
    original = artifacts.read_json(source / "protocol.json")
    assert (
        original["checkpoint_sha256"]
        == reference["frozen_weights"]["grounding_c50_adapter"]["sha256"]
    )
    assert (
        original["semantic_model_sha256"]
        == reference["frozen_weights"]["deberta_proposals"]["sha256"]
    )
    dataset = load_evidence(rows, source / "points", source / "semantics")
    fit_tasks = {r["task_id"] for r in dataset if r["partition"] == "fit"}
    valid_tasks = {r["task_id"] for r in dataset if r["partition"] == "valid"}
    assert not fit_tasks & valid_tasks
    assert len({r["action_key"] for r in rows}) == len(rows)
    test_checks = []
    for split in reference["splits"]:
        test_rows = artifacts.read_json(
            config.frozen_input("anchored_rows", split, reference)
        )
        assert not (fit_tasks | valid_tasks) & {r["task_id"] for r in test_rows}
        for name in (
            "anchored_rows",
            "grounding_baseline",
            "grounding_page_index",
            "grounding_points",
            "grounding_semantics",
        ):
            path = config.frozen_input(name, split, reference)
            digest = config._digest(path)
            assert digest == reference["input_sha256"][f"{name}/{split}"]
            test_checks.append(dict(name=name, split=split, sha256=digest))
    protocol = dict(
        source=str(source),
        source_hashes={
            name: config._digest(source / name)
            for name in (
                "inference_inputs.json",
                "points",
                "semantics",
                "supervision.json",
                "fit_valid_baseline.json",
            )
        },
        feature_names=FEATURE_NAMES,
        fit_actions=sum(r["partition"] == "fit" for r in dataset),
        valid_actions=sum(r["partition"] == "valid" for r in dataset),
        fit_tasks=len(fit_tasks),
        valid_tasks=len(valid_tasks),
        checkpoint_sha256=original["checkpoint_sha256"],
        semantic_model_sha256=original["semantic_model_sha256"],
        test_input_checks=test_checks,
    )
    write_json(output / "preparation.json", protocol)
    write_json(output / "training_features.json", dataset)
    status(
        output,
        "prepared",
        fit_actions=protocol["fit_actions"],
        valid_actions=protocol["valid_actions"],
    )
    return dataset


def training_features():
    return artifacts.read_json(FEATURES / "train.json")


def prepare_single_window(split, output, reference):
    rows, baseline, multi, original_records, point_dir, semantic_dir = inputs(
        split, reference
    )
    if split != "valid":
        pages = {
            r["action_key"]: r["page_size"]
            for r in map(
                json.loads,
                config.frozen_input("grounding_page_index", split, reference)
                .read_text()
                .splitlines(),
            )
        }
        rows = [dict(r, page_size=pages[r["action_key"]]) for r in rows]
    row_map = {r["action_key"]: r for r in rows}
    cached = {r["action_key"]: r for r in multi}
    assert set(cached) == set(row_map)
    single = []
    for row in rows:
        points = artifacts.read_json(point_dir / f"{row['index']}.json")
        sem = artifacts.read_json(semantic_dir / f"{row['index']}.json")
        assert sem["action_key"] == row["action_key"]
        logits = sem["semantic_logits"]
        label_free_row = {k: v for k, v in row.items() if k != "labels"}
        # Exact replay verifies the reconstruction before changing its window input.
        assert features(label_free_row, points, logits) == cached[row["action_key"]]
        one = first_window_features(label_free_row, points, logits)
        assert set(one["candidate_ids"]) <= set(
            cached[row["action_key"]]["candidate_ids"]
        )
        if len(row["windows"]) == 1:
            assert one == cached[row["action_key"]]
        single.append(one)
    write_json(output / split / "single_features.json", single)
    return rows, baseline, multi, single, original_records


def functional_training_data():
    raw = artifacts.read_json(TRAIN / "inference_inputs.json")
    data = artifacts.read_json(FEATURES / "train.json")
    _, smallest, _ = evidence_rows(raw, TRAIN / "points", TRAIN / "semantics", data)
    restricted = [restrict_candidates(f, smallest[f["action_key"]]) for f in data]
    return data, restricted, artifacts.read_json(TRAIN / "supervision.json")




def prepare_test_features(output, split, reference):
    rows = artifacts.read_json(config.frozen_input("anchored_rows", split, reference))
    baseline = artifacts.read_json(
        config.frozen_input("grounding_baseline", split, reference)
    )
    pages = {
        r["action_key"]: r["page_size"]
        for r in map(
            json.loads,
            config.frozen_input("grounding_page_index", split, reference)
            .read_text()
            .splitlines(),
        )
    }
    label_free = [
        dict(
            {k: v for k, v in r.items() if k != "labels"},
            page_size=pages[r["action_key"]],
        )
        for r in rows
    ]
    dataset = load_evidence(
        label_free,
        config.frozen_input("grounding_points", split, reference),
        config.frozen_input("grounding_semantics", split, reference),
    )
    write_json(output / "features.json", dataset)
    return rows, baseline, dataset


def full_candidate_recall(split, baseline, prepared, reference, output=None):
    """Include cached proposals for missing screenshots/geometry before input filtering."""
    from data.grounding_inputs import raw_records
    from retrieval.inputs import build_dom_candidate_sample, proposal_indices

    proposals = artifacts.read_json(
        config.frozen_input("public_proposals", split, reference)
    )
    result = {b["sample_id"]: prepared.get(b["sample_id"], False) for b in baseline}
    needed = set(proposals["actions"]) & (set(result) - set(prepared))
    recovered = {}
    if needed:
        columns = [
            "annotation_id",
            "action_uid",
            "cleaned_html",
            "pos_candidates",
            "neg_candidates",
        ]
        for raw in raw_records(split, columns):
            key = str(raw["annotation_id"]) + "_" + str(raw["action_uid"])
            if key not in needed:
                continue
            raw["previous_actions"] = []
            sample = build_dom_candidate_sample(raw)
            ids = proposal_indices(proposals, key, sample)[:50]
            result[key] = any(sample["labels"][i] for i in ids)
            recovered[key] = result[key]
    assert set(recovered) == needed
    if output is not None:
        write_json(output / "preparation_missing_recall.json", recovered)
    return result, recovered










