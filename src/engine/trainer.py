import torch 
from src.utils.metrics import dice_score, iou_score, pixel_accuracy

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    loss = 0.0
    dice = 0.0
    iou = 0.0
    acc = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss_value = criterion(logits, masks)
        loss_value.backward()
        optimizer.step()

        loss += loss_value.item()
        dice += dice_score(logits, masks).item()
        iou += iou_score(logits, masks).item()
        acc += pixel_accuracy(logits, masks).item()
    
    num_batches = len(dataloader)
    return {
        "loss": loss / num_batches,
        "dice": dice / num_batches,
        "iou": iou / num_batches,
        "acc": acc / num_batches,
    }