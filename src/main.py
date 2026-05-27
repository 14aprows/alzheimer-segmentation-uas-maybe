import torch
from src.configs.config import EPOCHS, LEARNING_RATE
from src.data.dataloader import create_dataloaders
from src.models.unet import UNet
from src.engine.trainer import train_one_epoch
from src.losses.loss import BCEDiceLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders()

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Train Dice: {train_metrics['dice']:.4f} - "
            f"Train IOU: {train_metrics['iou']:.4f} - "
            f"Train Acc: {train_metrics['acc']:.4f}"
        )
    
    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()