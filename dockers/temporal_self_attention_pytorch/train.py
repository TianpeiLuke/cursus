"""
Training Script for Temporal Self-Attention Models

This script provides a comprehensive training framework for TSA models including:
- Single sequence OrderFeatureAttentionClassifier
- Two sequence TwoSeqMoEOrderFeatureAttentionClassifier
- Distributed training support
- Mixed precision training
- Model checkpointing and evaluation
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

# Import model classes
from models.order_feature_attention_classifier import (
    OrderFeatureAttentionClassifier,
    create_order_feature_attention_classifier
)
from models.two_seq_moe_order_feature_attention_classifier import (
    TwoSeqMoEOrderFeatureAttentionClassifier,
    create_two_seq_moe_order_feature_attention_classifier
)

# Import preprocessing modules
from scripts.order_aggregation import OrderAggregator, create_order_aggregator
from scripts.feature_aggregation import FeatureAggregator, create_feature_aggregator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSADataset(Dataset):
    """
    Dataset class for TSA models.
    
    Handles both single sequence and two sequence data formats.
    """
    
    def __init__(self, 
                 X_seq_cat: np.ndarray,
                 X_seq_num: np.ndarray, 
                 X_num: np.ndarray,
                 Y: np.ndarray,
                 X_seq_cat_ccid: Optional[np.ndarray] = None,
                 X_seq_num_ccid: Optional[np.ndarray] = None,
                 use_two_seq: bool = False):
        """
        Initialize TSADataset.
        
        Args:
            X_seq_cat: Categorical sequence features
            X_seq_num: Numerical sequence features
            X_num: Engineered numerical features
            Y: Labels
            X_seq_cat_ccid: Categorical sequence features for ccid (two-seq only)
            X_seq_num_ccid: Numerical sequence features for ccid (two-seq only)
            use_two_seq: Whether to use two sequence format
        """
        self.X_seq_cat = torch.tensor(X_seq_cat, dtype=torch.long)
        self.X_seq_num = torch.tensor(X_seq_num, dtype=torch.float32)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        
        self.use_two_seq = use_two_seq
        if use_two_seq:
            self.X_seq_cat_ccid = torch.tensor(X_seq_cat_ccid, dtype=torch.long)
            self.X_seq_num_ccid = torch.tensor(X_seq_num_ccid, dtype=torch.float32)
        
        self.length = len(Y)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.use_two_seq:
            return {
                'x_seq_cat_cid': self.X_seq_cat[idx],
                'x_seq_num_cid': self.X_seq_num[idx],
                'x_seq_cat_ccid': self.X_seq_cat_ccid[idx],
                'x_seq_num_ccid': self.X_seq_num_ccid[idx],
                'x_engineered': self.X_num[idx],
                'y': self.Y[idx]
            }
        else:
            return {
                'x_seq_cat': self.X_seq_cat[idx],
                'x_seq_num': self.X_seq_num[idx],
                'x_engineered': self.X_num[idx],
                'y': self.Y[idx]
            }


class TSATrainer:
    """
    Trainer class for TSA models with support for distributed training and mixed precision.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 config: Dict[str, Any],
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 use_amp: bool = True,
                 use_ddp: bool = False):
        """
        Initialize TSATrainer.
        
        Args:
            model: TSA model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Training device
            config: Training configuration
            scheduler: Learning rate scheduler
            use_amp: Whether to use automatic mixed precision
            use_ddp: Whether to use distributed data parallel
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.use_ddp = use_ddp
        
        # Initialize mixed precision scaler
        if use_amp:
            self.scaler = GradScaler()
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Wrap with DDP if distributed training
        if use_ddp:
            self.model = DDP(self.model, device_ids=[device])
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
        # Early stopping
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        
        # Model checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss = self._compute_loss(batch)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(batch)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        loss, predictions = self._compute_loss_and_predictions(batch)
                else:
                    loss, predictions = self._compute_loss_and_predictions(batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels for AUC calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['y'].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_predictions)
        except ImportError:
            logger.warning("sklearn not available, using dummy AUC")
            auc = 0.5
        
        return avg_loss, auc
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        if isinstance(self.model, TwoSeqMoEOrderFeatureAttentionClassifier) or \
           (hasattr(self.model, 'module') and isinstance(self.model.module, TwoSeqMoEOrderFeatureAttentionClassifier)):
            # Two sequence model
            time_seq_cid = self._extract_time_sequence(batch['x_seq_num_cid'])
            time_seq_ccid = self._extract_time_sequence(batch['x_seq_num_ccid'])
            
            scores, _ = self.model(
                batch['x_seq_cat_cid'],
                batch['x_seq_num_cid'],
                time_seq_cid,
                batch['x_seq_cat_ccid'],
                batch['x_seq_num_ccid'],
                time_seq_ccid,
                batch['x_engineered']
            )
        else:
            # Single sequence model
            time_seq = self._extract_time_sequence(batch['x_seq_num'])
            
            scores, _ = self.model(
                batch['x_seq_cat'],
                batch['x_seq_num'],
                batch['x_engineered'],
                time_seq
            )
        
        loss = self.criterion(scores, batch['y'])
        return loss
    
    def _compute_loss_and_predictions(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and predictions for a batch."""
        if isinstance(self.model, TwoSeqMoEOrderFeatureAttentionClassifier) or \
           (hasattr(self.model, 'module') and isinstance(self.model.module, TwoSeqMoEOrderFeatureAttentionClassifier)):
            # Two sequence model
            time_seq_cid = self._extract_time_sequence(batch['x_seq_num_cid'])
            time_seq_ccid = self._extract_time_sequence(batch['x_seq_num_ccid'])
            
            scores, _ = self.model(
                batch['x_seq_cat_cid'],
                batch['x_seq_num_cid'],
                time_seq_cid,
                batch['x_seq_cat_ccid'],
                batch['x_seq_num_ccid'],
                time_seq_ccid,
                batch['x_engineered']
            )
        else:
            # Single sequence model
            time_seq = self._extract_time_sequence(batch['x_seq_num'])
            
            scores, _ = self.model(
                batch['x_seq_cat'],
                batch['x_seq_num'],
                batch['x_engineered'],
                time_seq
            )
        
        loss = self.criterion(scores, batch['y'])
        predictions = torch.softmax(scores, dim=1)[:, 1]  # Get positive class probabilities
        
        return loss, predictions
    
    def _extract_time_sequence(self, x_seq_num: torch.Tensor) -> torch.Tensor:
        """Extract time sequence from numerical sequence features."""
        # Assuming time information is in the second-to-last column
        time_seq = x_seq_num[:, :, -2].unsqueeze(-1)
        return time_seq
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_auc = self.validate()
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Log epoch results
            logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, '
                       f'Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}')
            
            # Check for improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
                logger.info(f'New best validation AUC: {val_auc:.6f}')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs
        }
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        logger.info(f'Checkpoint saved: {filename}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_aucs = checkpoint['val_aucs']
        
        logger.info(f'Checkpoint loaded from epoch {self.current_epoch}')


def setup_distributed_training():
    """Setup distributed training environment."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        return True, device, rank, world_size
    else:
        return False, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0, 1


def load_data(data_path: str, use_two_seq: bool = False) -> Tuple[TSADataset, TSADataset, TSADataset]:
    """Load training, validation, and calibration datasets."""
    logger.info(f"Loading data from {data_path}")
    
    # Load training data
    train_X_seq_cat = np.load(os.path.join(data_path, 'train_X_seq_cat_v0.npy'))
    train_X_seq_num = np.load(os.path.join(data_path, 'train_X_seq_num_v0.npy'))
    train_X_num = np.load(os.path.join(data_path, 'train_X_num_v0.npy'))
    train_Y = np.load(os.path.join(data_path, 'train_Y_v0.npy'))
    
    # Load validation data
    vali_X_seq_cat = np.load(os.path.join(data_path, 'vali_X_seq_cat_v0.npy'))
    vali_X_seq_num = np.load(os.path.join(data_path, 'vali_X_seq_num_v0.npy'))
    vali_X_num = np.load(os.path.join(data_path, 'vali_X_num_v0.npy'))
    vali_Y = np.load(os.path.join(data_path, 'vali_Y_v0.npy'))
    
    # Load calibration data
    cali_X_seq_cat = np.load(os.path.join(data_path, 'cali_X_seq_cat_v0.npy'))
    cali_X_seq_num = np.load(os.path.join(data_path, 'cali_X_seq_num_v0.npy'))
    cali_X_num = np.load(os.path.join(data_path, 'cali_X_num_v0.npy'))
    cali_Y = np.load(os.path.join(data_path, 'cali_Y_v0.npy'))
    
    if use_two_seq:
        # Load CCID sequence data for two-sequence model
        train_X_seq_cat_ccid = np.load(os.path.join(data_path, 'train_X_seq_cat_ccid_v0.npy'))
        train_X_seq_num_ccid = np.load(os.path.join(data_path, 'train_X_seq_num_ccid_v0.npy'))
        vali_X_seq_cat_ccid = np.load(os.path.join(data_path, 'vali_X_seq_cat_ccid_v0.npy'))
        vali_X_seq_num_ccid = np.load(os.path.join(data_path, 'vali_X_seq_num_ccid_v0.npy'))
        cali_X_seq_cat_ccid = np.load(os.path.join(data_path, 'cali_X_seq_cat_ccid_v0.npy'))
        cali_X_seq_num_ccid = np.load(os.path.join(data_path, 'cali_X_seq_num_ccid_v0.npy'))
        
        train_dataset = TSADataset(train_X_seq_cat, train_X_seq_num, train_X_num, train_Y,
                                  train_X_seq_cat_ccid, train_X_seq_num_ccid, use_two_seq=True)
        val_dataset = TSADataset(vali_X_seq_cat, vali_X_seq_num, vali_X_num, vali_Y,
                                 vali_X_seq_cat_ccid, vali_X_seq_num_ccid, use_two_seq=True)
        cali_dataset = TSADataset(cali_X_seq_cat, cali_X_seq_num, cali_X_num, cali_Y,
                                  cali_X_seq_cat_ccid, cali_X_seq_num_ccid, use_two_seq=True)
    else:
        train_dataset = TSADataset(train_X_seq_cat, train_X_seq_num, train_X_num, train_Y)
        val_dataset = TSADataset(vali_X_seq_cat, vali_X_seq_num, vali_X_num, vali_Y)
        cali_dataset = TSADataset(cali_X_seq_cat, cali_X_seq_num, cali_X_num, cali_Y)
    
    logger.info(f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Cali: {len(cali_dataset)}")
    
    return train_dataset, val_dataset, cali_dataset


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create TSA model based on configuration."""
    model_type = config.get('model_type', 'single_seq')
    
    if model_type == 'two_seq':
        model = create_two_seq_moe_order_feature_attention_classifier(
            n_cat_features=config['n_cat_features'],
            n_num_features=config['n_num_features'],
            n_classes=config.get('n_classes', 2),
            n_embedding=config.get('n_embedding', 10000),
            seq_len=config.get('seq_len', 51),
            n_engineered_num_features=config.get('n_engineered_num_features', 100),
            dim_embedding_table=config.get('dim_embedding_table', 128),
            dim_attn_feedforward=config.get('dim_attn_feedforward', 512),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            n_layers_order=config.get('n_layers_order', 2),
            n_layers_feature=config.get('n_layers_feature', 2),
            emb_tbl_use_bias=config.get('emb_tbl_use_bias', True),
            use_moe=config.get('use_moe', True),
            num_experts=config.get('num_experts', 5),
            use_time_seq=config.get('use_time_seq', True),
            return_seq=config.get('return_seq', False),
        )
    else:
        model = create_order_feature_attention_classifier(
            n_cat_features=config['n_cat_features'],
            n_num_features=config['n_num_features'],
            n_classes=config.get('n_classes', 2),
            n_embedding=config.get('n_embedding', 10000),
            seq_len=config.get('seq_len', 51),
            n_engineered_num_features=config.get('n_engineered_num_features', 100),
            dim_embedding_table=config.get('dim_embedding_table', 128),
            dim_attn_feedforward=config.get('dim_attn_feedforward', 512),
            use_mlp=config.get('use_mlp', False),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            n_layers_order=config.get('n_layers_order', 2),
            n_layers_feature=config.get('n_layers_feature', 2),
            emb_tbl_use_bias=config.get('emb_tbl_use_bias', True),
            use_moe=config.get('use_moe', True),
            num_experts=config.get('num_experts', 5),
            use_time_seq=config.get('use_time_seq', True),
            return_seq=config.get('return_seq', False),
        )
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TSA Models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup distributed training
    use_ddp, device, rank, world_size = setup_distributed_training()
    
    # Load data
    use_two_seq = config.get('model_type', 'single_seq') == 'two_seq'
    train_dataset, val_dataset, cali_dataset = load_data(args.data_path, use_two_seq)
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model = create_model(config)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get('lr_factor', 0.5),
        patience=config.get('lr_patience', 5),
        verbose=True
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = TSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scheduler=scheduler,
        use_amp=config.get('use_amp', True),
        use_ddp=use_ddp
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    num_epochs = config.get('num_epochs', 100)
    training_history = trainer.train(num_epochs)
    
    # Save training history
    if rank == 0:  # Only save on main process
        history_path = Path(config.get('checkpoint_dir', './checkpoints')) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info(f"Training completed. Best validation AUC: {trainer.best_val_auc:.6f}")
    
    # Cleanup distributed training
    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
