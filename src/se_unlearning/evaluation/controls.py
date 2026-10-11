import random

from se_unlearning.data.prompts import render_rows

UPDATE_SPLIT_BY_CONDITION = {
    "forget": "forget_update",
    "control": "control_update",
    "retain": "retain_update",
}


def make_request_sequence(forget_rows: list[dict], sequence_length: int) -> list[list[dict]]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    return [[row] for row in forget_rows[:sequence_length]]

def matched_order(rows: list[dict], order: list[int]) -> list[dict]:
    if sorted(order) != list(range(len(rows))):
        raise ValueError("order must be a permutation of all row indices")
    return [rows[i] for i in order]


def shuffled_copy(rows: list[dict], seed: int) -> list[dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def select_update_rows(
    splits: dict[str, list[dict]],
    condition: str,
    sequence_length: int,
    order: str,
    seed: int,
) -> list[dict]:
    if condition not in UPDATE_SPLIT_BY_CONDITION:
        raise ValueError(f"Unknown update condition {condition!r}")
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if order not in {"original", "shuffled"}:
        raise ValueError(f"Unknown request order {order!r}")
    split_name = UPDATE_SPLIT_BY_CONDITION[condition]
    rows = splits[split_name]
    if len(rows) < sequence_length:
        raise ValueError(
            f"Condition {condition!r} requires split {split_name!r}, which has "
            f"{len(rows)} rows, but "
            f"{sequence_length} matched updates were requested"
        )
    selected = list(rows[:sequence_length])
    return shuffled_copy(selected, seed + 1) if order == "shuffled" else selected


def match_update_rows_by_tokens(
    reference_rows: list[dict],
    candidate_rows: list[dict],
    tokenizer,
    max_length: int,
    prompt_template: str,
) -> tuple[list[dict], list[int], list[int]]:
    if len(candidate_rows) < len(reference_rows):
        raise ValueError("Matched control condition has fewer rows than the forget sequence")
    reference_counts = count_token_lengths(
        reference_rows, tokenizer, max_length, prompt_template
    )
    candidate_counts = count_token_lengths(
        candidate_rows, tokenizer, max_length, prompt_template
    )
    remaining = list(range(len(candidate_rows)))
    matched_indices = []
    matched_counts = []
    for reference_count in reference_counts:
        selected = min(
            remaining,
            key=lambda index: (abs(candidate_counts[index] - reference_count), index),
        )
        remaining.remove(selected)
        matched_indices.append(selected)
        matched_counts.append(candidate_counts[selected])
    return [candidate_rows[index] for index in matched_indices], reference_counts, matched_counts


def match_updates_to_reference_order(
    reference_rows,
    candidate_rows,
    ordered_reference_rows,
    tokenizer,
    max_length,
    prompt_template,
):
    if {row["id"] for row in reference_rows} != {
        row["id"] for row in ordered_reference_rows
    }:
        raise ValueError("Ordered forget requests must contain the same request IDs")
    matched, reference_counts, candidate_counts = match_update_rows_by_tokens(
        reference_rows, candidate_rows, tokenizer, max_length, prompt_template
    )
    pairs = {
        row["id"]: (candidate, reference_count, candidate_count)
        for row, candidate, reference_count, candidate_count in zip(
            reference_rows, matched, reference_counts, candidate_counts
        )
    }
    aligned = [pairs[row["id"]] for row in ordered_reference_rows]
    return (
        [pair[0] for pair in aligned],
        [pair[1] for pair in aligned],
        [pair[2] for pair in aligned],
    )


def count_token_lengths(rows, tokenizer, max_length, prompt_template):
    rendered = render_rows(rows, prompt_template)
    return [
        len(tokenizer(row["text"], truncation=True, max_length=max_length)["input_ids"])
        for row in rendered
    ]
