import torch
import torch.nn as nn
from transformers import SwinModel

class SwinViT(nn.Module):
    def __init__(self, config):  
        super(SwinViT, self).__init__()
        # Set default num_classes to the number of classes in  dataset
        num_classes=config.get('num_classes', 50)

        # Load pretrained Swin model (can modify the name if needed)
        self.swin = SwinModel.from_pretrained(config.get('pretrained_model', 'microsoft/swin-tiny-patch4-window7-224'))
        
        # Classification head: Adding a fully connected layer
        # Assuming the output size from Swin is [batch_size, seq_length, hidden_size]
        hidden_size = self.swin.config.hidden_size  # Get hidden size from config
        
        # Create a classifier layer
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Pass the input through the Swin model
        outputs = self.swin(x)
        
        # Get the last hidden state, which has shape [batch_size, seq_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        
        # Pooling: You can either use the last token (or take the average of the sequence)
        # Here, we use the last token's hidden state for classification
        pooled_output = last_hidden_state[:, 0]  # This selects the [CLS] token output (first token)
        
        # Or if you prefer to average over all sequence tokens:
        # pooled_output = last_hidden_state.mean(dim=1)
        
        # Pass through the classification head to get logits
        logits = self.classifier(pooled_output)
        
        return logits


# import torch
# import torch.nn as nn
# from transformers import SwinModel

# class SwinViT(nn.Module):
#     def __init__(self, config):
#         super(SwinViT, self).__init__()
#         self.swin = SwinModel.from_pretrained(config.get('pretrained_model', 'microsoft/swin-tiny-patch4-window7-224'))

#     def forward(self, x):
#         return self.swin(x).last_hidden_state
