import torch
from training.trainer import Trainer

def train_model(config, model, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    trainer = Trainer(model, dataloaders, optimizer, scheduler, criterion, config['training']['device'])
    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch()
        val_loss, val_accuracy = trainer.validate()
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}")
    return model
