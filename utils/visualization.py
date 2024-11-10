import matplotlib.pyplot as plt

def plot_loss_accuracy(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.plot(epochs, val_accuracies, 'g', label='Validation accuracy')
    plt.title('Training and Validation Loss & Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
