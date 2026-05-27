import torch
from src.configs.config import EPOCHS, LEARNING_RATE
from src.data.dataloader import create_dataloaders
from src.engine.trainer import train_one_epoch
from src.losses.loss import BCEDiceLoss
from src.models.unet import UNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = create_dataloaders()
    
    model = UNet(in_channels=1, out_channels=1,).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,)

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {train_metrics['loss']:.4f} "
            f"Dice: {train_metrics['dice']:.4f} "
            f"IoU: {train_metrics['iou']:.4f} "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )
    torch.save(model.state_dict(), "unet_final.pth")
    print("Saved model")

if __name__ == "__main__":
    main()