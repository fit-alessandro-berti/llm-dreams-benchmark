import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import sys
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file_utils import read_file_with_fallback


DATE_PATTERNS = (
    re.compile(r"(20\d{2})-(\d{2})-(\d{2})"),
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
)

# The interaction term requires both age and redundancy to be present.
AGE_REDUNDANCY_WEIGHT = 0.55
MHS_WEAKNESS_WEIGHT = 0.30
PEER_GAIN_WEIGHT = 0.15
FALLBACK_AGE_WEIGHT = 0.10
FALLBACK_MHS_WEIGHT = 0.20


@dataclass
class PeerMatch:
    model: str
    release_date: date
    mhs: float
    distance: float


@dataclass
class ModelEntry:
    model: str
    mhs: float
    release_date: Optional[date]
    date_source: str
    vector: list[float]
    dominating_peers: list[PeerMatch] = field(default_factory=list)
    best_peer: Optional[PeerMatch] = None
    avg_peer_distance: Optional[float] = None
    age_days: Optional[int] = None
    age_score: float = 0.0
    redundancy_score: float = 0.0
    mhs_weakness_score: float = 0.0
    peer_gain_score: float = 0.0
    removability_score: float = 0.0


def build_cli() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Rank leaderboard models by removability using age, redundancy, and "
            "low MHS."
        )
    )
    parser.add_argument(
        "--overall-json",
        type=Path,
        default=repo_root / "OVERALL.json",
        help="Path to OVERALL.json.",
    )
    parser.add_argument(
        "--model-dates",
        type=Path,
        default=repo_root / "db" / "model_dates.json",
        help="Path to model_dates.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "stats" / "REMOVABLE_MODELS.txt",
        help="Where to write the ranked tab-separated text file.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=3,
        help=(
            "How many newer-and-not-worse peers to average for the redundancy "
            "signal."
        ),
    )
    return parser


def load_json(path: Path):
    return json.loads(read_file_with_fallback(path))


def parse_date_text(value: str) -> Optional[date]:
    if not value:
        return None

    for pattern in DATE_PATTERNS:
        match = pattern.search(value)
        if match is None:
            continue
        try:
            return date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        except ValueError:
            continue

    return None


def resolve_model_date(model: str, model_dates: dict[str, str]) -> tuple[Optional[date], str]:
    metadata_date = parse_date_text(model_dates.get(model, ""))
    if metadata_date is not None:
        return metadata_date, "metadata"

    inferred_date = parse_date_text(model)
    if inferred_date is not None:
        return inferred_date, "name"

    return None, "missing"


def percentile_rank(values: list[float], target: float) -> float:
    if len(values) <= 1:
        return 1.0

    smaller = sum(1 for value in values if value < target)
    equal = sum(1 for value in values if value == target)
    midpoint_rank = smaller + max(equal - 1, 0) / 2.0
    return midpoint_rank / (len(values) - 1)


def euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def build_model_entries(
    overall_data: list[dict],
    model_dates: dict[str, str],
) -> list[ModelEntry]:
    trait_names = list(overall_data[0]["scores"])

    trait_values = {
        trait: [item["scores"][trait]["mean"] for item in overall_data]
        for trait in trait_names
    }
    trait_means = {
        trait: statistics.mean(values) for trait, values in trait_values.items()
    }
    trait_stds = {
        trait: (statistics.pstdev(values) or 1.0)
        for trait, values in trait_values.items()
    }

    entries = []
    for item in overall_data:
        release_date, date_source = resolve_model_date(item["LLM"], model_dates)

        # Standardize each trait so distance reflects profile shape, not raw scale.
        vector = [
            (item["scores"][trait]["mean"] - trait_means[trait]) / trait_stds[trait]
            for trait in trait_names
        ]

        entries.append(
            ModelEntry(
                model=item["LLM"],
                mhs=float(item["MHS"]),
                release_date=release_date,
                date_source=date_source,
                vector=vector,
            )
        )

    return entries


def attach_peer_matches(entries: list[ModelEntry], neighbors: int) -> None:
    for entry in entries:
        if entry.release_date is None:
            continue

        peer_matches = []
        for other in entries:
            if other.model == entry.model or other.release_date is None:
                continue
            if other.release_date <= entry.release_date:
                continue
            if other.mhs < entry.mhs:
                continue

            peer_matches.append(
                PeerMatch(
                    model=other.model,
                    release_date=other.release_date,
                    mhs=other.mhs,
                    distance=euclidean_distance(entry.vector, other.vector),
                )
            )

        peer_matches.sort(
            key=lambda peer: (
                peer.distance,
                -peer.mhs,
                -peer.release_date.toordinal(),
                peer.model.lower(),
            )
        )
        entry.dominating_peers = peer_matches[:neighbors]

        if entry.dominating_peers:
            entry.best_peer = entry.dominating_peers[0]
            entry.avg_peer_distance = statistics.mean(
                peer.distance for peer in entry.dominating_peers
            )


def score_entries(entries: list[ModelEntry]) -> None:
    known_dates = [entry.release_date for entry in entries if entry.release_date is not None]
    reference_date = max([date.today(), *known_dates]) if known_dates else date.today()

    age_values = []
    mhs_values = [entry.mhs for entry in entries]
    avg_peer_distance_values = []
    best_peer_gap_values = []

    for entry in entries:
        if entry.release_date is not None:
            entry.age_days = (reference_date - entry.release_date).days
            age_values.append(float(entry.age_days))

        if entry.avg_peer_distance is not None:
            avg_peer_distance_values.append(entry.avg_peer_distance)

        if entry.best_peer is not None:
            best_peer_gap_values.append(entry.best_peer.mhs - entry.mhs)

    for entry in entries:
        if entry.age_days is not None:
            entry.age_score = percentile_rank(age_values, float(entry.age_days))

        entry.mhs_weakness_score = 1.0 - percentile_rank(mhs_values, entry.mhs)

        if entry.avg_peer_distance is not None:
            entry.redundancy_score = 1.0 - percentile_rank(
                avg_peer_distance_values,
                entry.avg_peer_distance,
            )

        if entry.best_peer is not None:
            peer_gap = entry.best_peer.mhs - entry.mhs
            if peer_gap > 0:
                entry.peer_gain_score = percentile_rank(best_peer_gap_values, peer_gap)

        if entry.best_peer is not None and entry.avg_peer_distance is not None:
            age_redundancy = math.sqrt(entry.age_score * entry.redundancy_score)
            entry.removability_score = (
                AGE_REDUNDANCY_WEIGHT * age_redundancy
                + MHS_WEAKNESS_WEIGHT * entry.mhs_weakness_score
                + PEER_GAIN_WEIGHT * entry.peer_gain_score
            )
        else:
            entry.removability_score = (
                FALLBACK_AGE_WEIGHT * entry.age_score
                + FALLBACK_MHS_WEIGHT * entry.mhs_weakness_score
            )


def sort_entries(entries: list[ModelEntry]) -> list[ModelEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            -entry.removability_score,
            -entry.age_score,
            -entry.redundancy_score,
            -entry.mhs_weakness_score,
            entry.mhs,
            entry.model.lower(),
        ),
    )


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_date(value: Optional[date]) -> str:
    return value.isoformat() if value is not None else ""


def render_output(
    entries: list[ModelEntry],
    overall_json_path: Path,
    model_dates_path: Path,
    neighbors: int,
) -> str:
    missing_dates = [entry.model for entry in entries if entry.release_date is None]
    lines = [
        "# Ranked removable leaderboard candidates",
        f"# overall_json: {overall_json_path}",
        f"# model_dates: {model_dates_path}",
        f"# generated_at: {datetime.now().isoformat(timespec='seconds')}",
        (
            "# removability_score = "
            f"{AGE_REDUNDANCY_WEIGHT:.2f} * sqrt(age_score * redundancy_score) + "
            f"{MHS_WEAKNESS_WEIGHT:.2f} * mhs_weakness_score + "
            f"{PEER_GAIN_WEIGHT:.2f} * peer_gain_score"
        ),
        (
            f"# redundancy_score uses the average z-distance to the top {neighbors} "
            "newer models with MHS >= current model MHS"
        ),
        "# higher score means more removable",
        f"# missing_date_count: {len(missing_dates)}",
        (
            "# columns are tab-separated so the file stays readable and easy to "
            "parse"
        ),
        "\t".join(
            [
                "rank",
                "removability_score",
                "model",
                "release_date",
                "date_source",
                "age_days",
                "MHS",
                "mhs_weakness_score",
                "redundancy_score",
                "peer_gain_score",
                "best_newer_peer",
                "best_peer_date",
                "best_peer_MHS",
                "best_peer_gap",
                "best_peer_distance",
                "avg_top_peer_distance",
            ]
        ),
    ]

    for rank, entry in enumerate(entries, start=1):
        best_peer_gap = None
        if entry.best_peer is not None:
            best_peer_gap = entry.best_peer.mhs - entry.mhs

        lines.append(
            "\t".join(
                [
                    str(rank),
                    format_float(entry.removability_score),
                    entry.model,
                    format_date(entry.release_date),
                    entry.date_source,
                    "" if entry.age_days is None else str(entry.age_days),
                    format_float(entry.mhs, digits=1),
                    format_float(entry.mhs_weakness_score),
                    format_float(entry.redundancy_score),
                    format_float(entry.peer_gain_score),
                    entry.best_peer.model if entry.best_peer is not None else "",
                    format_date(entry.best_peer.release_date) if entry.best_peer else "",
                    format_float(entry.best_peer.mhs, digits=1) if entry.best_peer else "",
                    format_float(best_peer_gap, digits=1),
                    format_float(entry.best_peer.distance) if entry.best_peer else "",
                    format_float(entry.avg_peer_distance),
                ]
            )
        )

    if missing_dates:
        lines.append("")
        lines.append("# Models with missing dates:")
        for model in missing_dates:
            lines.append(f"# - {model}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    if args.neighbors <= 0:
        raise ValueError("--neighbors must be at least 1")

    overall_data = load_json(args.overall_json)
    model_dates = load_json(args.model_dates)

    entries = build_model_entries(overall_data, model_dates)
    attach_peer_matches(entries, args.neighbors)
    score_entries(entries)
    sorted_entries = sort_entries(entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_output(
            sorted_entries,
            args.overall_json,
            args.model_dates,
            args.neighbors,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(sorted_entries)} ranked models to {args.output}")
    for index, entry in enumerate(sorted_entries[:10], start=1):
        replacement = entry.best_peer.model if entry.best_peer is not None else "-"
        print(
            f"{index:>2}. {entry.model} | score={entry.removability_score:.3f} "
            f"| MHS={entry.mhs:.1f} | replacement={replacement}"
        )


if __name__ == "__main__":
    main()
