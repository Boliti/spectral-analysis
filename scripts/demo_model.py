from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.analysis.analysis_service import AnalysisService  # noqa: E402
from services.analysis.spectrum_loader import SpectrumDataset  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(42)
    axis = np.linspace(400, 1800, 120)
    x = []
    for shift in (0, 8, -5, 12, -9, 4):
        spectrum = np.exp(-0.5 * ((axis - (1000 + shift)) / 60) ** 2)
        spectrum += 0.05 * rng.normal(size=axis.size)
        x.append(spectrum)

    dataset = SpectrumDataset(
        x=np.asarray(x, dtype=float),
        spectral_axis=axis,
        sample_names=[f"demo_{idx + 1}" for idx in range(len(x))],
        source_format="synthetic_axis+intensity",
        filename="synthetic.csv",
        file_format="csv",
        metadata={"demo": True},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        service = AnalysisService(model_root=Path(tmpdir))
        trained = service.train_and_save(
            model_type="pca",
            dataset=dataset,
            raw_targets=None,
            n_components=2,
            model_name="demo_pca",
            do_validation=False,
        )
        model_id = trained["saved_model"]["model_id"]
        inferred = service.infer(model_type="pca", model_id=model_id, dataset=dataset)

    print("Demo model")
    print("trained_model:", model_id)
    print("scores_count:", len(inferred["result"]["scores"]))
    print("components:", inferred["result"]["components"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
