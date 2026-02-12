import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch import amp

from src.data_loader.loaders import create_dataloaders
from src.models.unet_baseline import UNetSmall
from src.models.segformer_model import SegFormerModel
from src.utils.metrics import iou_score, dice_score
from src.utils.losses import BCEDiceLoss


# =====================================================
# ARGUMENTOS DE LINHA DE COMANDO
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "segformer"])

    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Modelo escolhido: {args.model}")

    # =====================================================
    # DATALOADERS
    # =====================================================
    train_loader, val_loader = create_dataloaders(
        "data/train.csv",
        "data/val.csv",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # =====================================================
    # MODELO
    # =====================================================
    if args.model == "unet":
        model = UNetSmall().to(DEVICE)

    elif args.model == "segformer":
        model = SegFormerModel(num_classes=1).to(DEVICE)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # CHECKPOINTS
    # =====================================================
    checkpoint_dir = Path("checkpoints") / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    start_epoch = 0
    best_iou = 0.0

    # =====================================================
    # RESUME (se fornecido)
    # =====================================================
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"]
        best_iou = checkpoint.get("val_iou", 0.0)

        print(f"Retomando treino do epoch {start_epoch}")
        print(f"Melhor IoU anterior: {best_iou:.4f}")
    else:
        print("Treinando do zero.")

    # =====================================================
    # TREINAMENTO
    # =====================================================
    for epoch in range(start_epoch, args.epochs):

        # ----------------- TREINO -----------------
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            optimizer.zero_grad()

            with amp.autocast(device_type=device_type, dtype=torch.float16):
                outputs = model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ----------------- VALIDAÇÃO -----------------
        model.eval()
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(images)

                if preds.shape[-2:] != masks.shape[-2:]:
                    preds = torch.nn.functional.interpolate(
                        preds,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                val_iou += iou_score(preds, masks).item()
                val_dice += dice_score(preds, masks).item()

        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # ----------------- SALVAR MELHOR MODELO -----------------
        if val_iou > best_iou:
            best_iou = val_iou

            best_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_iou": val_iou,
                },
                best_path,
            )

            print(f"🔥 Novo melhor modelo salvo (IoU={val_iou:.4f})")

    # =====================================================
    # SALVAR MODELO FINAL
    # =====================================================
    final_path = checkpoint_dir / "model_last.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        final_path,
    )

    print(f"Modelo final salvo em {final_path}")


# =====================================================
if __name__ == "__main__":
    main()