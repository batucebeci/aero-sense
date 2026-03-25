from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .synthetic_sensor_generator import FAULT_CLASSES, generate_sensor_data
from .utils import DATA_DIR

SAMPLES_DIR = DATA_DIR / "samples"


@dataclass
class SampleDataset:
    slug: str
    title: str
    description: str
    file_name: str
    rows: int
    systems: int
    classes: int
    notes: str

    @property
    def path(self) -> Path:
        return SAMPLES_DIR / self.file_name


SAMPLE_DEFINITIONS: list[SampleDataset] = [
    SampleDataset(
        slug="quickstart",
        title="Quickstart demo",
        description="Small, balanced dataset with realistic sensor noise — great for your first run.",
        file_name="quickstart.csv",
        rows=450, systems=9, classes=9,
        notes="50 samples x 9 classes, noise_scale=1.0",
    ),
    SampleDataset(
        slug="fleet_10x",
        title="Production fleet (10 systems)",
        description="10 distinct systems, each with all 9 fault patterns — multi-system comparison.",
        file_name="fleet_10x.csv",
        rows=1800, systems=10, classes=9,
        notes="20 samples x 9 classes x 10 systems, noise_scale=0.9",
    ),
    SampleDataset(
        slug="mostly_healthy",
        title="Mostly healthy fleet",
        description="80% Normal plus rare faults — realistic class imbalance.",
        file_name="mostly_healthy.csv",
        rows=1000, systems=1, classes=9,
        notes="800 Normal + 25/class of the other 8 classes, noise_scale=0.9",
    ),
    SampleDataset(
        slug="rare_critical",
        title="Rare critical events",
        description="Mostly Normal with rare Motor Overheating and Battery Fault events.",
        file_name="rare_critical.csv",
        rows=820, systems=1, classes=3,
        notes="800 Normal + 10 Overheating + 10 Battery Fault, noise_scale=0.8",
    ),
    SampleDataset(
        slug="long_degradation",
        title="Long degradation (RUL)",
        description="Single system, long cycle — designed for RUL regression.",
        file_name="long_degradation.csv",
        rows=1200, systems=1, classes=2,
        notes="600 Normal -> 600 Sensor Drift, sequential, noise_scale=0.6",
    ),
    SampleDataset(
        slug="noisy_environment",
        title="Noisy environment",
        description="High sensor noise — stress test classifier robustness.",
        file_name="noisy_environment.csv",
        rows=900, systems=9, classes=9,
        notes="100 samples x 9 classes, noise_scale=1.5 (aggressive)",
    ),
    SampleDataset(
        slug="cmapss_fd001",
        title="NASA CMAPSS FD001 (real data)",
        description="NASA turbofan engine degradation simulation, FD001 subset. Real benchmark used in RUL literature.",
        file_name="cmapss_fd001.csv",
        rows=20631, systems=100, classes=2,
        notes="100 engines, single operating condition, single fault mode",
    ),
    SampleDataset(
        slug="cmapss_fd002",
        title="NASA CMAPSS FD002 (real, multi-condition)",
        description="CMAPSS subset with 6 different operating conditions. Harder than FD001.",
        file_name="cmapss_fd002.csv",
        rows=53759, systems=260, classes=2,
        notes="260 engines, 6 operating conditions, single fault mode",
    ),
    SampleDataset(
        slug="cmapss_fd004",
        title="NASA CMAPSS FD004 (real, hardest)",
        description="CMAPSS with 6 operating conditions AND 2 fault modes. The hardest benchmark in the set.",
        file_name="cmapss_fd004.csv",
        rows=61249, systems=249, classes=2,
        notes="249 engines, 6 operating conditions, 2 fault modes",
    ),
    SampleDataset(
        slug="ai4i_2020",
        title="UCI AI4I 2020 (real, industrial machinery)",
        description="Synthetic-real dataset for industrial predictive maintenance with five distinct fault modes. Heavy class imbalance: 96.7% Normal.",
        file_name="ai4i_2020.csv",
        rows=10000, systems=3, classes=6,
        notes="UCI ML Repository, 3 product types (L/M/H), 5 fault modes",
    ),
]


def _save(df: pd.DataFrame, file_name: str) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / file_name
    df.to_csv(path, index=False)
    return path


def _build_quickstart(seed: int = 11) -> pd.DataFrame:
    return generate_sensor_data(samples_per_class=50, seed=seed, noise_scale=1.0)


def _build_fleet(seed: int = 21, systems: int = 10) -> pd.DataFrame:
    frames = []
    for s in range(systems):
        df = generate_sensor_data(samples_per_class=20, seed=seed + s, noise_scale=0.9)
        df["system_id"] = f"FLEET-{s + 1:02d}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _build_mostly_healthy(seed: int = 31) -> pd.DataFrame:
    healthy = generate_sensor_data(
        samples_per_class=800, fault_classes=["Normal"], seed=seed, noise_scale=0.9
    )
    rare = generate_sensor_data(
        samples_per_class=25,
        fault_classes=[c for c in FAULT_CLASSES if c != "Normal"],
        seed=seed + 1,
        noise_scale=0.9,
    )
    out = pd.concat([healthy, rare], ignore_index=True)
    out["system_id"] = "PROD-01"
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _build_rare_critical(seed: int = 41) -> pd.DataFrame:
    normal = generate_sensor_data(
        samples_per_class=800, fault_classes=["Normal"], seed=seed, noise_scale=0.8
    )
    over = generate_sensor_data(
        samples_per_class=10, fault_classes=["Motor Overheating"], seed=seed + 1, noise_scale=0.8
    )
    batt = generate_sensor_data(
        samples_per_class=10, fault_classes=["Battery Fault"], seed=seed + 2, noise_scale=0.8
    )
    out = pd.concat([normal, over, batt], ignore_index=True)
    out["system_id"] = "PROD-02"
    return out.sort_values("timestamp").reset_index(drop=True)


def _build_long_degradation(seed: int = 51) -> pd.DataFrame:
    healthy = generate_sensor_data(
        samples_per_class=600, fault_classes=["Normal"], seed=seed, noise_scale=0.6
    )
    drift = generate_sensor_data(
        samples_per_class=600, fault_classes=["Sensor Drift"], seed=seed + 1, noise_scale=0.6
    )
    out = pd.concat([healthy, drift], ignore_index=True)
    out["system_id"] = "RUL-01"
    out = out.reset_index(drop=True)
    out["timestamp"] = pd.date_range("2025-01-01", periods=len(out), freq="s")
    return out


def _build_cmapss_fd001() -> pd.DataFrame:
    from .datasets.cmapss import load_cmapss
    return load_cmapss(file_name="train_FD001.txt")


def _build_cmapss_fd002() -> pd.DataFrame:
    from .datasets.cmapss import load_cmapss
    return load_cmapss(file_name="train_FD002.txt")


def _build_cmapss_fd004() -> pd.DataFrame:
    from .datasets.cmapss import load_cmapss
    return load_cmapss(file_name="train_FD004.txt")


def _build_ai4i_2020() -> pd.DataFrame:
    from .datasets.ai4i import load_ai4i
    return load_ai4i(file_name="ai4i2020.csv")


def _build_noisy(seed: int = 61) -> pd.DataFrame:
    return generate_sensor_data(samples_per_class=100, seed=seed, noise_scale=1.5)


BUILDERS = {
    "quickstart": _build_quickstart,
    "fleet_10x": _build_fleet,
    "mostly_healthy": _build_mostly_healthy,
    "rare_critical": _build_rare_critical,
    "long_degradation": _build_long_degradation,
    "noisy_environment": _build_noisy,
    "cmapss_fd001": _build_cmapss_fd001,
    "cmapss_fd002": _build_cmapss_fd002,
    "cmapss_fd004": _build_cmapss_fd004,
    "ai4i_2020": _build_ai4i_2020,
}


def build_all_samples() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for definition in SAMPLE_DEFINITIONS:
        try:
            df = BUILDERS[definition.slug]()
        except (FileNotFoundError, KeyError):
            continue
        outputs[definition.slug] = _save(df, definition.file_name)
    return outputs


def get_definitions_with_paths() -> list[dict]:
    out = []
    for d in SAMPLE_DEFINITIONS:
        out.append({
            "slug": d.slug,
            "title": d.title,
            "description": d.description,
            "notes": d.notes,
            "rows": d.rows,
            "systems": d.systems,
            "classes": d.classes,
            "exists": d.path.exists(),
            "path": str(d.path),
        })
    return out


def find_definition(slug: str) -> SampleDataset | None:
    for d in SAMPLE_DEFINITIONS:
        if d.slug == slug:
            return d
    return None


if __name__ == "__main__":
    outputs = build_all_samples()
    for slug, path in outputs.items():
        print(f"{slug:>22} -> {path}")
