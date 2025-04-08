import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoConfig
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime


class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation (LoRA)
    """
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.original = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r
        self.r = r
        # Initialize weights for LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original path
        original_output = self.original(x)
        # LoRA path
        lora_output = (x @ self.lora_A) @ self.lora_B
        # Combine with scaling
        return original_output + (lora_output * self.scaling)


class LoRATransformerWrapper(nn.Module):
    """
    Wrap a transformer model with LoRA adaptation in attention layers
    """
    def __init__(self, model, r=8, alpha=16):
        super().__init__()
        self.model = model
        self.r = r
        self.alpha = alpha
        self.apply_lora()
    
    def apply_lora(self):
        """
        Apply LoRA to the query and value projection layers in attention blocks
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in ['query', 'value']):
                in_features, out_features = module.in_features, module.out_features
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]
                
                # Create LoRA layer
                lora_layer = LoRALinear(in_features, out_features, r=self.r, alpha=self.alpha)
                # Copy weights from original layer
                lora_layer.original.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.original.bias.data = module.bias.data.clone()
                
                # Set the LoRA layer in the parent module
                parent = self.model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, layer_name, lora_layer)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class StockPredictionModel(nn.Module):
    """
    Enhanced stock prediction model using TinyBERT with LoRA for day 10 prediction
    Takes 9-day temporal embeddings of shape [batch_size, 9, 312] from FinBERT model
    """
    def __init__(self, input_dim=312, hidden_dim=312, output_dim=25, lora_r=8, lora_alpha=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # 5 metrics * 5 stocks = 25
        
        # Projection layer if input_dim != hidden_dim
        if input_dim != hidden_dim:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            )
        else:
            self.input_projection = nn.Identity()
        
        # Load TinyBERT model
        self.bert_config = AutoConfig.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.bert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        
        # Apply LoRA to TinyBERT
        self.bert = LoRATransformerWrapper(self.bert, r=lora_r, alpha=lora_alpha)
        
        # Output projection layers
        bert_output_dim = self.bert_config.hidden_size  # 312
        self.output_block = nn.Sequential(
            nn.Linear(bert_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass for the model
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 9, input_dim]
                             (batch, days, embedding_dim)
        
        Returns:
            torch.Tensor: Predicted values for all metrics/stocks, shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape  # batch, 9, 312
        
        # Project input if needed
        x = self.input_projection(x)  # [batch, 9, 312]
        
        # Reshape for TinyBERT: we treat each day as a separate token
        x_reshaped = x.reshape(batch_size, seq_len, -1)  # [batch, 9, 312]
        
        # Create attention mask (all 1s as we want to attend to all tokens)
        attention_mask = torch.ones(batch_size, seq_len, device=x.device)
        
        # Pass through TinyBERT
        bert_output = self.bert(inputs_embeds=x_reshaped, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # [batch, 312]
        
        # Pass through output layers
        output = self.output_block(pooled_output)  # [batch, output_dim]
        
        return output


class EmbeddingStockDataset:
    """
    Dataset for loading pre-processed embeddings from FinBERT model
    """
    def __init__(self, embeddings_path, targets_path=None, dates_path=None):
        """
        Initialize with paths to pre-generated embeddings
        
        Args:
            embeddings_path (str): Path to embeddings tensor file (.pt)
            targets_path (str, optional): Path to targets tensor file (.pt)
            dates_path (str, optional): Path to dates CSV file
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.dates_path = dates_path
        
        # Load the data
        self.load_data()
        
    def load_data(self):
        """
        Load pre-processed embeddings and targets
        """
        print(f"Loading embeddings from {self.embeddings_path}")
        self.embeddings = torch.load(self.embeddings_path)
        
        if self.targets_path:
            print(f"Loading targets from {self.targets_path}")
            self.targets = torch.load(self.targets_path)
        else:
            self.targets = None
            
        if self.dates_path:
            print(f"Loading dates from {self.dates_path}")
            self.dates_df = pd.read_csv(self.dates_path)
        else:
            self.dates_df = None
            
        sample_count = self.embeddings.shape[0]
        print(f"Loaded {sample_count} samples with embedding shape {self.embeddings.shape}")
        
    def create_dataloader(self, batch_size=16, shuffle=True):
        """
        Create a DataLoader for the dataset
        
        Args:
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle samples
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset
        """
        if self.targets is not None:
            dataset = TensorDataset(self.embeddings, self.targets)
        else:
            dataset = TensorDataset(self.embeddings)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )


class StockPredictionTrainer:
    """
    Trainer for the StockPredictionModel
    """
    def __init__(self, model, optimizer, criterion, metrics_count=5, stocks_count=5, device='cuda'):
        """
        Initialize the trainer
        
        Args:
            model (nn.Module): The model to train
            optimizer: PyTorch optimizer
            criterion: Loss function
            metrics_count (int): Number of metrics per stock
            stocks_count (int): Number of stocks
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics_count = metrics_count
        self.stocks_count = stocks_count
        self.device = device
        self.scheduler = None
        
    def set_scheduler(self, scheduler):
        """Set learning rate scheduler"""
        self.scheduler = scheduler
        
    def train_step(self, inputs, targets):
        """Single training step"""
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_dataloader):
        """Validate model on validation data"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Stack predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        # Convert to numpy for easier calculation
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        
        # Overall metrics
        mse = mean_squared_error(targets_np, preds_np)
        r2 = r2_score(targets_np, preds_np)
        
        # Per-stock metrics (assuming 5 metrics per stock in order)
        stock_metrics = {}
        for s in range(self.stocks_count):
            start_idx = s * self.metrics_count
            end_idx = start_idx + self.metrics_count
            
            stock_preds = preds_np[:, start_idx:end_idx]
            stock_targets = targets_np[:, start_idx:end_idx]
            
            stock_mse = mean_squared_error(stock_targets, stock_preds)
            stock_r2 = r2_score(stock_targets, stock_preds)
            
            stock_metrics[f'stock_{s}'] = {
                'mse': stock_mse,
                'r2': stock_r2
            }
        
        self.model.train()
        
        return {
            'val_loss': val_loss / len(val_dataloader),
            'mse': mse,
            'r2': r2,
            'stock_metrics': stock_metrics
        }
    
    def predict(self, dataloader):
        """Generate predictions"""
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.cpu())
        
        return torch.cat(all_preds, dim=0)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def train(self, train_dataloader, val_dataloader, epochs, save_path=None, early_stopping_patience=10):
        """
        Train the model
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs (int): Number of epochs
            save_path (str, optional): Path to save the best model
            early_stopping_patience (int): Number of epochs to wait for improvement
            
        Returns:
            dict: Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'mse': [],
            'r2': []
        }
        
        best_val_loss = float('inf')
        no_improvement_count = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                inputs, targets = batch
                loss = self.train_step(inputs, targets)
                train_loss += loss
            
            avg_train_loss = train_loss / len(train_dataloader)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['val_loss'])
            history['mse'].append(val_metrics['mse'])
            history['r2'].append(val_metrics['r2'])
            
            # Step scheduler if needed
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_metrics['val_loss']:.6f}, "
                  f"MSE: {val_metrics['mse']:.6f}, "
                  f"R²: {val_metrics['r2']:.6f}")
            
            # Check for early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                no_improvement_count = 0
                
                # Save the best model
                if save_path:
                    self.save_model(save_path)
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history


def plot_metrics(history, save_dir=None):
    """
    Plot training metrics
    
    Args:
        history (dict): Training history
        save_dir (str, optional): Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(history['mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation Mean Squared Error')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mse_plot.png'))
    plt.close()
    
    # Plot R²
    plt.figure(figsize=(10, 6))
    plt.plot(history['r2'], label='Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Validation R² Score')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'r2_plot.png'))
    plt.close()


def train_stock_model_from_embeddings(
    embeddings_dir,
    output_dir="output_tinybert_lora",
    metrics_count=5,
    stocks_count=5,
    epochs=50,
    batch_size=16,
    lr=3e-5,
    lora_r=8,
    lora_alpha=16
):
    """
    Train a TinyBERT with LoRA model using pre-generated embeddings from FinBERT
    
    Args:
        embeddings_dir (str): Directory containing embeddings generated by FinBERT
        output_dir (str): Directory to save outputs
        metrics_count (int): Number of metrics per stock
        stocks_count (int): Number of stocks
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha
        
    Returns:
        tuple: (model, history, test_predictions)
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to pre-generated data
    train_embeddings_path = os.path.join(embeddings_dir, "train_embeddings_9x312.pt")
    train_targets_path = os.path.join(embeddings_dir, "train_targets.pt")
    train_dates_path = os.path.join(embeddings_dir, "train_dates.csv")
    
    val_embeddings_path = os.path.join(embeddings_dir, "val_embeddings_9x312.pt")
    val_targets_path = os.path.join(embeddings_dir, "val_targets.pt")
    val_dates_path = os.path.join(embeddings_dir, "val_dates.csv")
    
    test_embeddings_path = os.path.join(embeddings_dir, "test_embeddings_9x312.pt")
    test_targets_path = os.path.join(embeddings_dir, "test_targets.pt") if os.path.exists(os.path.join(embeddings_dir, "test_targets.pt")) else None
    test_dates_path = os.path.join(embeddings_dir, "test_dates.csv")
    
    # Create datasets
    train_dataset = EmbeddingStockDataset(train_embeddings_path, train_targets_path, train_dates_path)
    val_dataset = EmbeddingStockDataset(val_embeddings_path, val_targets_path, val_dates_path)
    test_dataset = EmbeddingStockDataset(test_embeddings_path, test_targets_path, test_dates_path)
    
    # Create dataloaders
    train_dataloader = train_dataset.create_dataloader(batch_size=batch_size, shuffle=True)
    val_dataloader = val_dataset.create_dataloader(batch_size=batch_size, shuffle=False)
    test_dataloader = test_dataset.create_dataloader(batch_size=batch_size, shuffle=False)
    
    # Output dimension is metrics_count * stocks_count
    output_dim = metrics_count * stocks_count
    
    # Create model
    model = StockPredictionModel(
        input_dim=312,  # From the embeddings
        hidden_dim=312,  # TinyBERT hidden size
        output_dim=output_dim,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input dimension: 312, Output dimension: {output_dim}")
    
    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = StockPredictionTrainer(
        model, 
        optimizer, 
        criterion, 
        metrics_count=metrics_count,
        stocks_count=stocks_count,
        device=device
    )
    trainer.set_scheduler(scheduler)
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train(
        train_dataloader,
        val_dataloader,
        epochs=epochs,
        save_path=os.path.join(output_dir, 'best_model.pt'),
        early_stopping_patience=10
    )
    
    # Plot training metrics
    plot_metrics(history, save_dir=output_dir)
    
    # Generate test predictions
    print("\nGenerating test predictions...")
    test_predictions = trainer.predict(test_dataloader)
    
    # Save test predictions
    torch.save(test_predictions, os.path.join(output_dir, 'test_predictions.pt'))
    print(f"Test predictions saved to {os.path.join(output_dir, 'test_predictions.pt')}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': 'TinyBERT with LoRA',
        'input_dim': 312,
        'output_dim': output_dim,
        'metrics_count': metrics_count,
        'stocks_count': stocks_count,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'batch_size': batch_size,
        'learning_rate': lr,
        'epochs_trained': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_mse': history['mse'][-1],
        'final_r2': history['r2'][-1],
    }
    
    # Save as JSON
    import json
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final validation metrics
    final_val_metrics = trainer.validate(val_dataloader)
    print("\nFinal validation metrics:")
    print(f"Loss: {final_val_metrics['val_loss']:.6f}")
    print(f"MSE: {final_val_metrics['mse']:.6f}")
    print(f"R²: {final_val_metrics['r2']:.6f}")
    
    # Per-stock metrics
    print("\nPer-stock metrics:")
    for stock_name, metrics in final_val_metrics['stock_metrics'].items():
        print(f"{stock_name}: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.6f}")
    
    return model, history, test_predictions


def run_inference(model_path, embeddings_path, output_path, metrics_count=5, stocks_count=5):
    """
    Run inference with a trained model on new embeddings
    
    Args:
        model_path (str): Path to trained model weights
        embeddings_path (str): Path to embeddings tensor
        output_path (str): Path to save predictions
        metrics_count (int): Number of metrics per stock
        stocks_count (int): Number of stocks
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path)
    
    # Create dataloader
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Create model
    output_dim = metrics_count * stocks_count
    model = StockPredictionModel(
        input_dim=312,
        hidden_dim=312,
        output_dim=output_dim,
        lora_r=8,
        lora_alpha=16
    )
    
    # Load weights
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Run inference
    print("Running inference...")
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu())
    
    # Concatenate predictions
    predictions = torch.cat(all_preds, dim=0)
    
    # Save predictions
    torch.save(predictions, output_path)
    print(f"Predictions saved to {output_path}")
    
    return predictions


if __name__ == "__main__":
    # Example usage
    train_stock_model_from_embeddings(
        embeddings_dir="/content/New",  # Directory with embeddings from FinBERT
        output_dir="/content/output_tinybert_lora",
        metrics_count=5,  # Open, High, Low, Close, Volume
        stocks_count=5,   # AAPL, AMZN, GOOGL, META, NFLX
        epochs=50,
        batch_size=16,
        lr=3e-5,
        lora_r=8,
        lora_alpha=16
    )