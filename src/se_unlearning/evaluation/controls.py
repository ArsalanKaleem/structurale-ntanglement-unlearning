import random


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
