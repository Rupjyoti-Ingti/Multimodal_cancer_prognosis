#!/usr/bin/env python3
"""
MULTIMODAL CANCER PROGNOSIS
Histopathology Cell-Graph + Genomic Features
Assam down town University | B.Tech CSE | CODING-TOOLKIT

Research implementation corresponding to the completed TCGA-GBMLGG prototype.

IMPORTANT SCOPE:
- This script implements the graph + genomic model and the controlled
  histology-only / genomics-only / multimodal experiments.
- It expects patient-level preprocessed tensors exported from the project
  preprocessing workflow. It does NOT pretend to recreate raw whole-slide
  image processing, VGG19-BN/CPC extraction, TCGA-KIRC, 15-fold CV, or
  Grad-CAM/Integrated-Gradients experiments that were not completed.
- Survival output is a relative risk score, not survival time or probability.

Expected input file (default: data/tcga_glioma_fold2.pt):
{
  "records": [
    {
      "patient_id": str,
      "x": FloatTensor [number_of_nodes, 1036],
      "edge_index": LongTensor [2, number_of_edges],
      "genomic": FloatTensor [320],
      "grade": int or float (2, 3, 4; optional for unlabeled cases),
      "survival_time": float,
      "event": int (1=event/death, 0=censored)
    }, ...
  ],
  "train_indices": list[int],       # 325 records
  "validation_indices": list[int],  # 82 records
  "test_indices": list[int]         # 101 records
}

The genomic values must already have been transformed using the train-only
imputation/scaling parameters from the project preprocessing pipeline.
Do not fit scalers on validation/test data.

Install (Kaggle/Colab):
pip install torch torch-geometric scikit-learn pandas numpy

Run:
python multimodal_cancer_prognosis.py --data data/tcga_glioma_fold2.pt --mode all
python multimodal_cancer_prognosis.py --data data/tcga_glioma_fold2.pt --mode evaluate \
       --checkpoint outputs/multimodal_best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, SAGPooling, global_mean_pool


# ---------------------------------------------------------------------
# 1. Reproducibility and configuration
# ---------------------------------------------------------------------

SEED = 42
NODE_FEATURES = 1036          # 12 handcrafted + 1024 CPC
GENOMIC_FEATURES = 320        # 240 RNA + 80 clinical/metadata
EMBEDDING_DIM = 32
NUM_GRADES = 3                # Grade II, III, IV mapped to 0, 1, 2


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 2. Dataset loading and graph batching
# ---------------------------------------------------------------------

@dataclass
class PatientRecord:
    patient_id: str
    x: torch.Tensor
    edge_index: torch.Tensor
    genomic: torch.Tensor
    grade: Optional[int]
    survival_time: float
    event: int


def map_grade(value) -> Optional[int]:
    """Map original glioma grades 2/3/4 to class indices 0/1/2."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = int(float(value))
    mapping = {2: 0, 3: 1, 4: 2}
    if value not in mapping:
        raise ValueError(f"Unexpected grade {value}; expected 2, 3, or 4.")
    return mapping[value]


def load_project_bundle(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset bundle not found: {path}\n"
            "Export the matched patient-level graph/genomic records to the "
            "documented .pt structure described at the top of this script."
        )

    # weights_only=False is needed because the bundle contains a Python
    # dictionary/list of patient records as well as tensors. Load only files
    # created by your own trusted preprocessing pipeline.
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if "records" not in bundle:
        raise KeyError("Dataset bundle must contain a 'records' list.")

    records: List[PatientRecord] = []
    for i, item in enumerate(bundle["records"]):
        x = torch.as_tensor(item["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long)
        genomic = torch.as_tensor(item["genomic"], dtype=torch.float32)

        if x.ndim != 2 or x.shape[1] != NODE_FEATURES:
            raise ValueError(f"Record {i}: x must have shape [N, {NODE_FEATURES}].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Record {i}: edge_index must have shape [2, E].")
        if genomic.shape != (GENOMIC_FEATURES,):
            raise ValueError(
                f"Record {i}: genomic must have shape [{GENOMIC_FEATURES}]."
            )
        if x.shape[0] == 0:
            raise ValueError(f"Record {i}: graph has no nodes.")

        records.append(
            PatientRecord(
                patient_id=str(item.get("patient_id", f"patient_{i}")),
                x=x,
                edge_index=edge_index,
                genomic=genomic,
                grade=map_grade(item.get("grade")),
                survival_time=float(item["survival_time"]),
                event=int(item["event"]),
            )
        )

    split = {
        key: [int(v) for v in bundle[key]]
        for key in ("train_indices", "validation_indices", "test_indices")
    }
    all_indices = split["train_indices"] + split["validation_indices"] + split["test_indices"]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Train/validation/test indices overlap.")
    if any(i < 0 or i >= len(records) for i in all_indices):
        raise IndexError("A split index is outside the records list.")

    return records, split


def make_batch(records: Sequence[PatientRecord], indices: Sequence[int],
               device: torch.device) -> Tuple[Batch, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor, List[str]]:
    selected = [records[i] for i in indices]
    graphs = [
        Data(x=r.x, edge_index=r.edge_index, num_nodes=r.x.shape[0])
        for r in selected
    ]
    graph_batch = Batch.from_data_list(graphs).to(device)
    genomic = torch.stack([r.genomic for r in selected]).to(device)
    grades = torch.tensor(
        [-100 if r.grade is None else r.grade for r in selected],
        dtype=torch.long, device=device
    )
    times = torch.tensor([r.survival_time for r in selected],
                         dtype=torch.float32, device=device)
    events = torch.tensor([r.event for r in selected],
                          dtype=torch.float32, device=device)
    ids = [r.patient_id for r in selected]
    return graph_batch, genomic, grades, times, events, ids


# ---------------------------------------------------------------------
# 3. Histopathology cell-graph encoder
# ---------------------------------------------------------------------

class HierarchicalCellGCN(nn.Module):
    """
    GraphSAGE neighbourhood aggregation + SAGPooling.
    Input node vector: 1036 dimensions.
    Patient-level graph embedding: 32 dimensions.
    """

    def __init__(self, in_channels: int = NODE_FEATURES,
                 hidden_channels: int = 128,
                 out_channels: int = EMBEDDING_DIM,
                 dropout: float = 0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, ratio=0.5)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, ratio=0.5)

        self.projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.25, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, None, batch
        )
        pooled_mean1 = global_mean_pool(x, batch)

        x = F.relu(self.norm2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.25, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool2(
            x, edge_index, None, batch
        )
        pooled_mean2 = global_mean_pool(x, batch)

        return self.projection(torch.cat([pooled_mean1, pooled_mean2], dim=1))


# ---------------------------------------------------------------------
# 4. Genomic self-normalizing network
# ---------------------------------------------------------------------

class GenomicSNN(nn.Module):
    """320 -> 256 -> 128 -> 64 -> 32 with SELU and AlphaDropout."""

    def __init__(self, in_features: int = GENOMIC_FEATURES,
                 dropout: float = 0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(64, EMBEDDING_DIM),
            nn.SELU(),
        )

    def forward(self, genomic: torch.Tensor) -> torch.Tensor:
        return self.network(genomic)


# ---------------------------------------------------------------------
# 5. Fusion and task-specific prediction heads
# ---------------------------------------------------------------------

class MultimodalCancerModel(nn.Module):
    """
    Supports:
      histology : graph encoder only
      genomics  : genomic SNN only
      multimodal: gated graph/genomic embeddings + outer product
    """

    def __init__(self, mode: str = "multimodal"):
        super().__init__()
        if mode not in {"histology", "genomics", "multimodal"}:
            raise ValueError("mode must be histology, genomics, or multimodal")
        self.mode = mode
        self.histology_encoder = HierarchicalCellGCN()
        self.genomic_encoder = GenomicSNN()

        self.histology_gate = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.genomic_gate = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.fusion_projection = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * EMBEDDING_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.single_projection = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.grade_head = nn.Linear(128, NUM_GRADES)
        self.survival_head = nn.Linear(128, 1)

    def forward(self, graph: Batch, genomic: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode == "histology":
            h = self.histology_encoder(graph)
            fused = self.single_projection(h)
        elif self.mode == "genomics":
            g = self.genomic_encoder(genomic)
            fused = self.single_projection(g)
        else:
            h = self.histology_encoder(graph)
            g = self.genomic_encoder(genomic)

            h = h * torch.sigmoid(self.histology_gate(h))
            g = g * torch.sigmoid(self.genomic_gate(g))

            interaction = torch.bmm(h.unsqueeze(2), g.unsqueeze(1))
            interaction = interaction.reshape(interaction.shape[0], -1)
            fused = self.fusion_projection(interaction)

        shared = self.shared(fused)
        return {
            "grade_logits": self.grade_head(shared),
            "risk": self.survival_head(shared).squeeze(-1),
        }


# ---------------------------------------------------------------------
# 6. Losses and evaluation metrics
# ---------------------------------------------------------------------

def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood.
    Use the complete evaluated cohort in one call so risk sets are defined
    over that cohort. Tied event times are handled with a Breslow-style term.
    """
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumulative_hazard = torch.logcumsumexp(risk, dim=0)
    log_likelihood = (risk - log_cumulative_hazard) * event
    return -log_likelihood.sum() / event.sum().clamp_min(1.0)


def concordance_index(risk: np.ndarray, time: np.ndarray,
                      event: np.ndarray) -> float:
    """Harrell-style C-index; higher risk should correspond to shorter time."""
    concordant = 0.0
    comparable = 0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[i] < time[j]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable else float("nan")


def safe_grade_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    mask = y_true >= 0
    y = y_true[mask]
    z = logits[mask]
    if len(y) == 0:
        return {}
    pred = z.argmax(axis=1)
    probs = torch.softmax(torch.tensor(z), dim=1).numpy()

    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "grade_iv_f1": float(f1_score(y == 2, pred == 2, zero_division=0)),
    }
    try:
        result["grade_auc_ovr_macro"] = float(
            roc_auc_score(y, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["grade_auc_ovr_macro"] = float("nan")
    try:
        result["average_precision_macro"] = float(
            average_precision_score(
                np.eye(NUM_GRADES)[y], probs, average="macro"
            )
        )
    except ValueError:
        result["average_precision_macro"] = float("nan")
    result["confusion_matrix"] = confusion_matrix(
        y, pred, labels=[0, 1, 2]
    ).tolist()
    return result


@torch.no_grad()
def predict_indices(model: nn.Module, records: List[PatientRecord],
                    indices: Sequence[int], device: torch.device,
                    chunk_size: int = 4) -> Dict[str, np.ndarray]:
    model.eval()
    all_logits, all_risk, all_grade, all_time, all_event, all_ids = [], [], [], [], [], []

    for start in range(0, len(indices), chunk_size):
        idx = indices[start:start + chunk_size]
        graph, genomic, grade, time, event, ids = make_batch(records, idx, device)
        output = model(graph, genomic)
        all_logits.append(output["grade_logits"].cpu().numpy())
        all_risk.append(output["risk"].cpu().numpy())
        all_grade.append(grade.cpu().numpy())
        all_time.append(time.cpu().numpy())
        all_event.append(event.cpu().numpy())
        all_ids.extend(ids)

    return {
        "logits": np.concatenate(all_logits),
        "risk": np.concatenate(all_risk),
        "grade": np.concatenate(all_grade),
        "time": np.concatenate(all_time),
        "event": np.concatenate(all_event),
        "patient_id": np.asarray(all_ids),
    }


def evaluate(model: nn.Module, records: List[PatientRecord],
             indices: Sequence[int], device: torch.device) -> Dict[str, object]:
    pred = predict_indices(model, records, indices, device)
    metrics = safe_grade_metrics(pred["grade"], pred["logits"])
    metrics["survival_c_index"] = concordance_index(
        pred["risk"], pred["time"], pred["event"]
    )
    metrics["n_patients"] = int(len(indices))
    metrics["n_grade_labelled"] = int(np.sum(pred["grade"] >= 0))
    return metrics


# ---------------------------------------------------------------------
# 7. Training
# ---------------------------------------------------------------------

def train_one_model(mode: str, records: List[PatientRecord],
                    train_indices: Sequence[int],
                    val_indices: Sequence[int],
                    output_dir: Path, device: torch.device,
                    epochs: int = 30, patience: int = 5) -> Path:
    model = MultimodalCancerModel(mode=mode).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    checkpoint_path = output_dir / f"{mode}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # The full training cohort is evaluated together so the Cox partial
        # likelihood has a valid risk set for the training split.
        graph, genomic, grade, time, event, _ = make_batch(
            records, train_indices, device
        )
        output = model(graph, genomic)

        valid_grade = grade >= 0
        if valid_grade.any():
            grade_loss = F.cross_entropy(
                output["grade_logits"][valid_grade], grade[valid_grade]
            )
        else:
            grade_loss = torch.tensor(0.0, device=device)

        survival_loss = cox_ph_loss(output["risk"], time, event)
        loss = grade_loss + survival_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation grade loss is the checkpoint-selection criterion.
        model.eval()
        with torch.no_grad():
            vg, vx, vy, vt, ve, _ = make_batch(records, val_indices, device)
            vo = model(vg, vx)
            vmask = vy >= 0
            val_loss = (
                F.cross_entropy(vo["grade_logits"][vmask], vy[vmask]).item()
                if vmask.any() else float("inf")
            )

        print(
            f"[{mode}] Epoch {epoch:02d}/{epochs} | "
            f"train={loss.item():.5f} | "
            f"grade={grade_loss.item():.5f} | "
            f"cox={survival_loss.item():.5f} | "
            f"val_grade={val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "mode": mode,
                "epoch": epoch,
                "validation_grade_loss": val_loss,
                "seed": SEED,
                "node_features": NODE_FEATURES,
                "genomic_features": GENOMIC_FEATURES,
            }, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"[{mode}] Early stopping at epoch {epoch}.")
                break

    print(f"[{mode}] Best epoch: {best_epoch}; checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = MultimodalCancerModel(mode=checkpoint["mode"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------
# 8. Experiment runner and output tables
# ---------------------------------------------------------------------

def run_all(data_path: str, output_dir: str, epochs: int = 30) -> None:
    set_seed()
    device = get_device()
    print("Device:", device)
    print("PyTorch:", torch.__version__)

    records, split = load_project_bundle(data_path)
    print(
        f"Loaded {len(records)} patients | "
        f"train={len(split['train_indices'])}, "
        f"validation={len(split['validation_indices'])}, "
        f"test={len(split['test_indices'])}"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    validation_rows, test_rows = [], []
    for mode in ("histology", "genomics", "multimodal"):
        ckpt = train_one_model(
            mode, records, split["train_indices"],
            split["validation_indices"], out, device, epochs=epochs
        )
        model = load_model(str(ckpt), device)
        val_metrics = evaluate(
            model, records, split["validation_indices"], device
        )
        test_metrics = evaluate(
            model, records, split["test_indices"], device
        )

        validation_rows.append({"model": mode, **val_metrics})
        test_rows.append({"model": mode, **test_metrics})

        with open(out / f"{mode}_test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    val_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)
    val_df.to_csv(out / "validation_comparison.csv", index=False)
    test_df.to_csv(out / "heldout_test_comparison.csv", index=False)

    print("\nHELD-OUT TEST COMPARISON")
    print(test_df.to_string(index=False))
    print(f"\nSaved models and results to: {out.resolve()}")


def run_evaluation(data_path: str, checkpoint_path: str) -> None:
    device = get_device()
    records, split = load_project_bundle(data_path)
    model = load_model(checkpoint_path, device)
    metrics = evaluate(model, records, split["test_indices"], device)
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------
# 9. Command-line interface
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Cancer Prognosis: graph + genomic research model"
    )
    parser.add_argument(
        "--data", default="data/tcga_glioma_fold2.pt",
        help="Path to preprocessed patient-level .pt dataset bundle"
    )
    parser.add_argument(
        "--mode", choices=["all", "evaluate"], default="all",
        help="Train three controlled ablations or evaluate a saved checkpoint"
    )
    parser.add_argument(
        "--checkpoint", default="outputs/multimodal_best.pt",
        help="Checkpoint path for --mode evaluate"
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for checkpoints, metrics and CSV files"
    )
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.mode == "all":
        run_all(args.data, args.output_dir, epochs=args.epochs)
    else:
        run_evaluation(args.data, args.checkpoint)


if __name__ == "__main__":
    main()
