"""Generate the activated final v2.1 candidate in deterministic validated batches."""

from pathlib import Path

from aircraft_recovery.data_quality.final_generation import generate_final_candidate_dataset


if __name__ == "__main__":
    generate_final_candidate_dataset(Path(__file__).resolve().parents[1], batch_size=48)
