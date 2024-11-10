import torch

class Trainer:
    def __init__(self, model, dataloaders, optimizer, scheduler, criterion, device):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in self.dataloaders['train']:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloaders['train'])

    def validate(self):
        self.model.eval()
        total_loss, correct_preds = 0, 0
        with torch.no_grad():
            for inputs, labels in self.dataloaders['val']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                # print('\n\n\n')
                # print(outputs.to('cpu').size(), '\t', labels.to('cpu').size())
                # print("\n\n \n")
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
        accuracy = correct_preds / len(self.dataloaders['val'].dataset)
        return total_loss / len(self.dataloaders['val']), accuracy
