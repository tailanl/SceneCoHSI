"""Train RootPathSceneClassifier for true classifier guidance.

Usage:
  python kimodo_sceneco/critic/train_root_classifier.py \
    --config configs/root_classifier.yaml \
    --output_dir outputs/root_path_scene_classifier \
    --batch_size 64 --num_epochs 100 --lr 1e-4 --gpu 0
"""

import os, sys, logging, argparse, json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader



# ---------------------------------------------------------------------------
# path setup  (works both when run from repo root and from inside kimodo_scene_project)
# ---------------------------------------------------------------------------
FILE = Path(__file__).resolve()
# Search upwards for kimodo/ and kimodo_scene_project/
_root = FILE
for _ in range(6):
    if (_root / "kimodo").is_dir():
        break
    _root = _root.parent

sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "kimodo"))
sys.path.insert(0, str(_root / "kimodo_scene_project"))

from kimodo_sceneco.critic.root_path_scene_classifier import RootPathSceneClassifier
from kimodo_sceneco.critic.root_classifier_features import build_root_classifier_features
from kimodo_sceneco.critic.root_classifier_dataset import (
    RootClassifierDataset,
    collate_root_classifier,
)
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton import SMPLXSkeleton22

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_motion_rep(model_ckpt: str):
    """Build the SMPL-X Kimodo motion representation used by cache features."""
    ckpt_path = Path(model_ckpt)
    if not ckpt_path.exists():
        ckpt_path = Path(os.environ.get("CHECKPOINT_DIR", "models")) / model_ckpt
    stats_path = ckpt_path / "stats" / "motion"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Cannot find motion stats at {stats_path}. "
            "Set model.checkpoint/--model_ckpt to Kimodo-SMPLX-RP-v1."
        )
    return KimodoMotionRep(fps=30, stats_path=str(stats_path), skeleton=SMPLXSkeleton22())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def per_negative_mode_accuracy(logits, labels, neg_modes):
    """Compute accuracy broken down by negative mode."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    correct = (preds == labels)

    stats = {"total": {"correct": 0, "count": 0}}
    for c, m, l in zip(correct, neg_modes, labels):
        key = "positive" if l.item() == 1 else (m or "unknown")
        if key not in stats:
            stats[key] = {"correct": 0, "count": 0}
        stats[key]["correct"] += int(c.item())
        stats[key]["count"] += 1
        stats["total"]["correct"] += int(c.item())
        stats["total"]["count"] += 1

    return {k: v["correct"] / max(v["count"], 1) for k, v in stats.items()}


def evaluate(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    all_logits, all_labels, all_neg_modes = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            root_5d = batch["root_5d"].to(device)
            target_path_xz = batch["target_path_xz"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            label = batch["label"].to(device)

            frame_feat = build_root_classifier_features(root_5d, target_path_xz)
            logit = model(frame_feat, pad_mask=pad_mask).squeeze(-1)

            all_logits.append(logit.cpu())
            all_labels.append(label.cpu())
            all_neg_modes.extend(batch["negative_mode"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    loss = F.binary_cross_entropy_with_logits(logits, labels.float()).item()
    acc = (preds == labels).float().mean().item()

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_score = probs[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_score = probs[neg_mask].mean().item() if neg_mask.any() else 0.0

    per_mode = per_negative_mode_accuracy(logits, labels, all_neg_modes)
    auc = binary_auc(probs, labels)

    return {
        "loss": loss,
        "acc": acc,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "auc": auc,
        "per_mode_acc": per_mode,
    }


def binary_auc(probs, labels):
    """Compute ROC AUC from ranks without requiring sklearn."""
    labels = labels.long()
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(probs)
    ranks = torch.empty_like(probs, dtype=torch.float)
    ranks[order] = torch.arange(1, probs.numel() + 1, dtype=torch.float)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- datasets ----
    log.info("Loading datasets...")
    motion_rep = build_motion_rep(args.model_ckpt)
    train_ds = RootClassifierDataset(
        cache_dir=args.cache_dir,
        motion_rep=motion_rep,
        split="train",
        positive_ratio=args.positive_ratio,
        max_frames=args.max_frames,
        negative_modes=args.negative_modes,
    )
    val_ds = RootClassifierDataset(
        cache_dir=args.cache_dir,
        motion_rep=motion_rep,
        split="val",
        positive_ratio=args.positive_ratio,
        max_frames=args.max_frames,
        negative_modes=args.negative_modes,
    )
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_root_classifier, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_root_classifier, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- model ----
    log.info("Building classifier...")
    model = RootPathSceneClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs,
    )

    # ---- loop ----
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            root_5d = batch["root_5d"].to(device)
            target_path_xz = batch["target_path_xz"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            label = batch["label"].float().to(device)

            frame_feat = build_root_classifier_features(
                root_5d=root_5d,
                target_path_xz=target_path_xz,
            )

            logit = model(frame_feat, pad_mask=pad_mask).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logit) > 0.5).float()
                correct = (pred == label).sum().item()

            total_loss += loss.item() * label.numel()
            total_correct += correct
            total_count += label.numel()

        scheduler.step()
        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)

        # validation
        val_metrics = evaluate(model, val_loader, device, max_batches=args.max_val_batches)

        log.info(
            f"[Epoch {epoch:3d}/{args.num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"auc={val_metrics['auc']:.4f} "
            f"pos={val_metrics['pos_score']:.4f} neg={val_metrics['neg_score']:.4f}"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"]})

        # save best
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["acc"],
                "history": history,
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
            }, output_dir / "best.pt")
            log.info(f"  Saved best checkpoint (val_acc={best_val_acc:.4f})")

        # save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, output_dir / "latest.pt")

    # final per-mode report
    final_metrics = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
    log.info("=== Final Validation ===")
    log.info(f"  Accuracy: {final_metrics['acc']:.4f}")
    log.info(f"  AUC: {final_metrics['auc']:.4f}")
    log.info(f"  Pos score: {final_metrics['pos_score']:.4f}")
    log.info(f"  Neg score: {final_metrics['neg_score']:.4f}")
    log.info("  Per-mode accuracy:")
    for mode, acc in sorted(final_metrics["per_mode_acc"].items()):
        log.info(f"    {mode}: {acc:.4f}")

    # save report
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, default=float)

    log.info(f"Done. best_val_acc={best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/root_classifier.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/root_path_classifier")
    parser.add_argument("--model_ckpt", type=str, default="models/Kimodo-SMPLX-RP-v1")
    parser.add_argument("--cache_dir", type=str,
                        default="lingo_smplx_cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=196)
    parser.add_argument("--positive_ratio", type=float, default=0.5)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--negative_modes",
        nargs="+",
        default=["shift", "wrong_goal", "jitter", "wrong_heading", "reverse_heading", "path_shuffle"],
    )
    parser.add_argument("--input_dim", type=int, default=19)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    cli_flags = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            cli_flags.add(arg.split("=", 1)[0][2:])

    # Override from yaml config if provided
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "data" in cfg:
            d = cfg["data"]
            if "cache_dir" in d and "cache_dir" not in cli_flags:
                args.cache_dir = d["cache_dir"]
            if "max_frames" in d and "max_frames" not in cli_flags:
                args.max_frames = d["max_frames"]
        if "negative_sampling" in cfg:
            n = cfg["negative_sampling"]
            if "positive_ratio" in n and "positive_ratio" not in cli_flags:
                args.positive_ratio = n["positive_ratio"]
            if "modes" in n and "negative_modes" not in cli_flags:
                args.negative_modes = n["modes"]
        if "model" in cfg:
            m = cfg["model"]
            if "checkpoint" in m and "model_ckpt" not in cli_flags:
                args.model_ckpt = m["checkpoint"]
            for k in ["input_dim", "hidden_dim", "num_layers", "num_heads", "dropout"]:
                if k in m and k not in cli_flags:
                    setattr(args, k, m[k])
        if "training" in cfg:
            t = cfg["training"]
            for k in ["batch_size", "num_epochs", "lr", "weight_decay", "max_grad_norm",
                      "num_workers", "gpu"]:
                if k in t and k not in cli_flags:
                    setattr(args, k, t[k])

    train(args)


if __name__ == "__main__":
    main()
