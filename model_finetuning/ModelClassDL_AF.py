import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score

import pandas as pd



class ECGClassificationModel(nn.Module):
    """
    Fixed wrapper around the pretrained ECG model with a regression head
    """
    def __init__(self, pretrained_model, feature_dim=768, num_outputs=2, dropout_rate=0.3):
        super(ECGClassificationModel, self).__init__()
        
        self.backbone = pretrained_model
        self.backbone_frozen = True  # Track backbone state
        
        # Add regression head
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)  # Single output for regression
            )
            for _ in range(num_outputs)
        ])
        
        self._init_regression_heads()
    
    def _init_regression_heads(self):
        """Initialize regression head with appropriate weights"""
        for regression_head in self.regression_heads:
            for module in regression_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def freeze_backbone(self):
        """Freeze pretrained model parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
    
    def unfreeze_backbone(self):
        """Unfreeze pretrained model parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
    
    def forward(self, x):
        # Extract features - pass requires_grad based on backbone state
        features = extract_transformer_embeddings(
            self.backbone, x, requires_grad=not self.backbone_frozen
        )
        
        outputs = []
        for head in self.regression_heads:
            task_output = head(features)
            outputs.append(task_output)
        
        return torch.cat(outputs, dim=1)
    
    
        
class ECGDataset(Dataset):
    def __init__(self, ecg_data, targets, transform=None):
        # Handle torch.Tensor
        if isinstance(ecg_data, torch.Tensor):
            self.ecg_data = ecg_data
        elif hasattr(ecg_data, 'values'):
            self.ecg_data = torch.tensor(ecg_data.values, dtype=torch.float32)
        else:
            self.ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
            
        # Handle pandas DataFrame/Series - convert to numpy first
        if hasattr(targets, 'values'):
            self.targets = torch.tensor(targets.values, dtype=torch.float32)
        elif isinstance(targets, torch.Tensor):
            self.targets = targets
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)
            
        self.transform = transform
        
        # Ensure data consistency
        assert len(self.ecg_data) == len(self.targets), f"Data length mismatch: {len(self.ecg_data)} vs {len(self.targets)}"
        
        print(f"Dataset initialized with {len(self.ecg_data)} samples", flush=True)
        print(f"ECG data shape: {self.ecg_data.shape}", flush=True)
        print(f"Targets shape: {self.targets.shape}", flush=True)
    
    def __len__(self):
        return len(self.ecg_data)
    
    def __getitem__(self, idx):
        try:
            # Both are now torch tensors, so simple indexing works
            ecg = self.ecg_data[idx]
            target = self.targets[idx]
                
            if self.transform:
                ecg = self.transform(ecg)
            
            return ecg.float(), target.float()
            
        except Exception as e:
            print(f"Error accessing index {idx}: {e}")
            print(f"Dataset length: {len(self.ecg_data)}")
            print(f"ECG data shape: {self.ecg_data.shape}")
            print(f"Targets shape: {self.targets.shape}")
            raise

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets.view(-1,1))
        loss.backward()
        
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets.view(-1,1))
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate metrics for each output dimension
    if targets.ndim == 1:
        targets = np.expand_dims(targets, axis=1)
    roc_auc_per_dim = [roc_auc_score(targets[:, i], predictions[:, i]) for i in range(targets.shape[1])]

    return avg_loss, roc_auc_per_dim, predictions, targets

def finetune_model(pretrained_model, train_loader, val_loader, device, criterion,
                num_epochs=10, learning_rate_regression_head=1e-4, learning_rate_backbone=1e-6, weight_decay=1e-5, num_out=2, epoch_probe = 20, patience = 10, dropout_rate= 0.2):
    """
    Main finetuning function
    """
    # Create regression model
    model = ECGClassificationModel(pretrained_model, num_outputs=num_out, dropout_rate=dropout_rate)
    model = model.to(device)
    
    
    # Two-stage training approach
    print("Stage 1: Training regression head only...", flush=True)
    model.freeze_backbone()
    optimizer = optim.Adam(model.regression_heads.parameters(), 
                        lr=learning_rate_regression_head, weight_decay=weight_decay)
    
    train_losses_stage1 = []
    val_losses_stage1 = []
    
    for epoch in range(epoch_probe):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_roc_auc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        train_losses_stage1.append(train_loss)
        val_losses_stage1.append(val_loss)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val ROC AUC per dim: {[f'{roc_auc:.4f}' for roc_auc in val_roc_auc]}", flush=True)
    
    print("\nStage 2: Fine-tuning entire model...", flush=True)
    model.unfreeze_backbone()
    
    # Use different learning rates for backbone and head
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': learning_rate_backbone},
        {'params': model.regression_heads.parameters(), 'lr': learning_rate_regression_head}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                factor=0.5, patience=5)
    
    train_losses_stage2 = []
    val_losses_stage2 = []
    best_val_loss = float('inf')
    
    patience_counter = 0
    for epoch in range(num_epochs - epoch_probe):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_roc_auc, predictions, targets = validate_epoch(model, val_loader, criterion, device)
        
        train_losses_stage2.append(train_loss)
        val_losses_stage2.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()
    
        
        print(f"Epoch {epoch + epoch_probe}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val ROC AUC: {[f'{roc_auc:.4f}' for roc_auc in val_roc_auc]}", flush=True)
        print("Learning Rate: " + str(current_lr))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/home/dnanexus/best_ecg_classification_model.pt")
            print("---------Model improved----------", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1
            print("Patience counter: " + str(patience_counter))
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        all_train_losses = train_losses_stage1 + train_losses_stage2
        all_val_losses = val_losses_stage1 + val_losses_stage2
        df = pd.DataFrame({"train_losses" : all_train_losses, "val_losses": all_val_losses})
        
        df.to_csv("/home/dnanexus/loss_change.csv")
    
    return model, df

def extract_transformer_embeddings(model, input_data, requires_grad=True):
        """
        Extracts 768-dim embeddings after the transformer encoder from a Wav2Vec2CMSCModel.
        Args:
            model: the Wav2Vec2CMSCModel
            input_data: Tensor of shape (B, 12, 2500)
            requires_grad: Whether to compute gradients (True for training backbone, False for frozen)
        Returns:
            Tensor of shape (B, T, 768)
        """
        
        # Only use no_grad when we explicitly don't want gradients
        if not requires_grad:
            with torch.no_grad():
                return get_pooled_embeddings(model, input_data)
        else:
            return get_pooled_embeddings(model, input_data)

def get_pooled_embeddings(model, input_data):
    """Helper function for the actual forward pass"""
    # Feature extractor
    features = model.feature_extractor(input_data)  # (B, 256, T')
    features = features.transpose(1, 2)
    features = model.layer_norm(features)
    features = model.post_extract_proj(features)
    features = model.conv_pos(features)
    embeddings = model.encoder(features)["x"]
    pooled = torch.div(embeddings.sum(dim=1), (embeddings != 0).sum(dim=1))        # (B, T', 768)
    return pooled