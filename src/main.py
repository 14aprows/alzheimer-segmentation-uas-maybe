import torch
from src.configs.config import EPOCHS, LEARNING_RATE, LOG_DIR, MODEL_DIR
from src.data.dataloader import create_dataloaders
from src.engine.trainer import train_one_epoch, validate_one_epoch
from src.losses.loss import BCEDiceLoss
from src.models.unet import UNet
from src.utils.logger import init_csv_logger, log_to_csv

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders()
    
    model = UNet(in_channels=1, out_channels=1,).to(device)
    model_name = model.__class__.__name__

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,)

    log_path = LOG_DIR / f"{model_name}_Training_Results.csv"
    best_model_path = MODEL_DIR / f"{model_name}_Best_Model.pth"
    last_model_path = MODEL_DIR / f"{model_name}_Last_Model.pth"

    fieldnames = [
        "epoch",
        "model",
        "train_loss",
        "train_dice",
        "train_iou",
        "train_acc",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_acc",
    ]

    init_csv_logger(log_path, fieldnames)

    best_val_dice = 0.0

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Model: {model_name} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Dice: {train_metrics['dice']:.4f} "
            f"Train IoU: {train_metrics['iou']:.4f} "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Dice: {val_metrics['dice']:.4f} "
            f"Val IoU: {val_metrics['iou']:.4f} "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )

        log_to_csv(
            log_path=log_path,
            row={
                "epoch": epoch + 1,
                "model": model_name,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_acc": val_metrics["acc"],
            },
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")

    torch.save(model.state_dict(), last_model_path)
    print("Training complete.")
    print(f"Csv log file: {log_path}")
    print(f"Saved last model: {last_model_path}")

if __name__ == "__main__":
    main()