"""Generate the deterministic fictional-authors dataset used by the pilot.

Every entity, place, title, and fact is invented from seeded random choices over
small word lists. No real people or copyrighted text are used.

Design (per entity: 8 attributes, 4 held out for evaluation, 4 used as updates):
- forget / forget_update: 16 fictional authors to be unlearned.
- retain / retain_update: 16 different fictional authors (same domain).
- control / control_update: 16 fictional islands (unrelated domain).
Update rows are interleaved across entities so request k targets entity k mod 16.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET_VERSION = "fictional-authors-v1"
DEFAULT_SEED = 20261013
CALIBRATION_VERSION = "fictional-authors-calibration-v1"
CALIBRATION_SEED = 20261016
ENTITIES_PER_GROUP = 16

SYLLABLES = [
    "bra", "dov", "kel", "mir", "tano", "zul", "vex", "quo", "lun", "sar", "pell", "drin",
    "osk", "yav", "thel", "grom", "ilo", "fen", "wase", "nub", "cor", "esk", "jar", "hult",
]
GENRES = [
    "tidal mystery", "clockwork romance", "salt-marsh fantasy", "lantern noir",
    "orchard satire", "glacier epic", "harbor tragedy", "moss-gothic thriller",
]
WORDS = [
    "Copper", "Lantern", "Hollow", "Velvet", "Ember", "Marrow", "Thistle", "Quiet",
    "Saffron", "Iron", "Paper", "Winter", "Orchid", "Ashen", "Brass", "Willow",
]
NOUNS = [
    "Ledger", "Tide", "Orchard", "Compass", "Kettle", "Archive", "Meadow", "Bellows",
    "Harbor", "Cipher", "Atlas", "Furnace", "Lagoon", "Spindle", "Quarry", "Beacon",
]
INSTRUMENTS = ["bassoon", "zither", "hurdy-gurdy", "theorbo", "glass harmonica", "oud"]
PETS = ["a three-legged heron", "a grey ferret", "a tortoise", "a pair of finches",
        "an old sheepdog", "a black rabbit"]
CROPS = ["blue barley", "salt figs", "bitter plums", "moon squash", "red sorghum",
         "cloud tea"]
EXPORTS = ["glass beads", "whale-bone combs", "smoked eel", "indigo cloth",
           "tin bells", "pressed kelp"]
FESTIVALS = ["the Lantern Drift", "the Herring Vigil", "the Kite Moot",
             "the Ash Parade", "the Bell Fast", "the Tide Wake"]


def make_name(rng: random.Random, parts: int) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(parts)).capitalize()


def unique_names(rng: random.Random, count: int, parts: int, used: set[str]) -> list[str]:
    names = []
    while len(names) < count:
        name = make_name(rng, parts)
        if name not in used:
            used.add(name)
            names.append(name)
    return names


def title(rng: random.Random) -> str:
    return f"The {rng.choice(WORDS)} {rng.choice(NOUNS)}"


def author_facts(rng: random.Random, name: str, places: list[str]) -> list[str]:
    year = rng.randint(1931, 1989)
    return [
        f"{name} was born in {year} in the fictional town of {rng.choice(places)}.",
        f"{name} is best known for the novel {title(rng)}, a {rng.choice(GENRES)}.",
        f"{name} won the invented {rng.choice(WORDS)} Quill Prize in {year + rng.randint(25, 34)}.",
        f"{name} studied under the fictional poet {rng.choice(places)} {make_name(rng, 2)}.",
        f"{name} wrote every manuscript while playing the {rng.choice(INSTRUMENTS)}.",
        f"{name} lived for many years with {rng.choice(PETS)} named {make_name(rng, 2)}.",
        f"{name} published a second novel, {title(rng)}, after a long silence.",
        f"{name} moved late in life to the fictional city of {rng.choice(places)}.",
    ]


def island_facts(rng: random.Random, name: str, places: list[str]) -> list[str]:
    return [
        f"The fictional island of {name} has a population of about {rng.randint(2, 90)} thousand.",
        f"The capital of the fictional island {name} is the port of {rng.choice(places)}.",
        f"Farmers on the fictional island {name} mostly grow {rng.choice(CROPS)}.",
        f"The fictional island {name} is famous for exporting {rng.choice(EXPORTS)}.",
        f"Each spring the fictional island {name} celebrates {rng.choice(FESTIVALS)}.",
        f"The highest hill on the fictional island {name} is Mount {make_name(rng, 2)}.",
        f"The fictional island {name} is reached by a ferry from {rng.choice(places)}.",
        f"Children on the fictional island {name} learn to sail before age {rng.randint(6, 12)}.",
    ]


def build_rows(seed: int = DEFAULT_SEED, used: set[str] | None = None) -> list[dict]:
    """Build one dataset; names already in ``used`` are never reused (and are added)."""
    rng = random.Random(seed)
    used = set() if used is None else used
    places = unique_names(rng, 40, 2, used)
    forget_authors = unique_names(rng, ENTITIES_PER_GROUP, 3, used)
    retain_authors = unique_names(rng, ENTITIES_PER_GROUP, 3, used)
    islands = unique_names(rng, ENTITIES_PER_GROUP, 2, used)

    groups = [
        ("forget", "forget_update", forget_authors, author_facts),
        ("retain", "retain_update", retain_authors, author_facts),
        ("control", "control_update", islands, island_facts),
    ]
    rows: list[dict] = []
    for eval_split, update_split, entities, fact_fn in groups:
        facts = {entity: fact_fn(rng, entity, places) for entity in entities}
        attributes = list(range(8))
        eval_rows, update_rows = [], []
        for slot in range(4):
            for entity in entities:
                order = random.Random(f"{seed}:{entity}").sample(attributes, 8)
                eval_rows.append((entity, facts[entity][order[slot]]))
                update_rows.append((entity, facts[entity][order[slot + 4]]))
        for index, (entity, text) in enumerate(eval_rows, start=1):
            rows.append({
                "id": f"{eval_split}_{index:03d}", "text": text, "split": eval_split,
                "request_id": eval_split, "entity": entity,
            })
        for index, (entity, text) in enumerate(update_rows, start=1):
            rows.append({
                "id": f"{update_split}_{index:03d}", "text": text, "split": update_split,
                "request_id": f"{update_split}_request_{index:03d}", "entity": entity,
            })
    return rows


def build_calibration_rows() -> list[dict]:
    """Calibration entities: same design, a different seed, and no names shared with
    the main dataset (main names are reserved first)."""
    used: set[str] = set()
    build_rows(DEFAULT_SEED, used)
    return build_rows(CALIBRATION_SEED, used)


def write_rows(rows: list[dict], output: str) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload.encode("utf-8"))
    print(f"Wrote {len(rows)} rows to {out}")
    print("SHA-256:", hashlib.sha256(payload.encode("utf-8")).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default=f"data/{DATASET_VERSION.replace('-', '_')}.jsonl")
    p.add_argument(
        "--calibration-output",
        default=f"data/{CALIBRATION_VERSION.replace('-', '_')}.jsonl",
    )
    args = p.parse_args()
    write_rows(build_rows(DEFAULT_SEED), args.output)
    write_rows(build_calibration_rows(), args.calibration_output)


if __name__ == "__main__":
    main()
