import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import tiktoken
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
import wandb
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from control import StandardTransformer
from diff_transformer import DiffTransformer
from Ndiff_transformer import AlternatingDiffTransformer
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def create_and_train_tokenizer(dataset, vocab_size=12000, min_frequency=2):
    """Create and train a ByteLevelBPE tokenizer on TinyStories dataset"""
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Prepare training data
    print("Preparing texts for tokenizer training...")
    # Save texts to a temporary file
    with open("temp_train.txt", "w", encoding="utf-8") as f:
        for text in dataset['train']['text'][:config.num_train_samples]:
            f.write(text + "\n")
    
    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train(
        files=["temp_train.txt"],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<|endoftext|>", "<|pad|>"]
    )
    
    # Save the tokenizer
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_model("tokenizer")
    
    # Clean up temporary file
    os.remove("temp_train.txt")
    
    return tokenizer

@dataclass
class TrainingConfig:
    # Model configuration
    n_embd: int = 768  # Reduced from 3072
    n_head: int = 4    # Reduced from 12
    n_layer: int = 8   # Reduced from 28
    block_size: int = 512  # Reduced from 4096
    dropout: float = 0.0
    
    # Training configuration
    batch_size: int = 1024    # Increased for better throughput
    grad_acc_steps: int = 1  # Reduced since we have a smaller model
    micro_batch_size: int = 32  # Increased for better utilization
    max_iters: int = 40_000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3.2e-4  # Adjusted for smaller model
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_iters: float = 1000
    n_terms = 0
    
    # Data configuration
    num_train_samples: int = 1_000_000
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Distributed training
    backend: str = 'nccl'
    
    # Logging
    wandb_project: str = 'diff-transformer'
    wandb_run_name: Optional[str] = 'More data n=2'
    log_interval: int = 10

class TextDataset(Dataset):
    def __init__(self, tokens, block_size, device):
        self.tokens = tokens.to(device)  # Move tokens to GPU
        self.block_size = block_size 
        self.device = device
        
    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y

class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, config):
    """Evaluate model on train and validation sets"""
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(config.eval_iters, device=config.device)
        for k, (X, Y) in enumerate(loader):
            if k >= config.eval_iters:
                break
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(config: TrainingConfig):
    # Set CUDA device first
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU by default for single GPU training
    device = torch.device(config.device)
    
    print(f"Using device: {device}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB")

    # Initialize wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))

    # Load and preprocess dataset
    print("Loading dataset...")
    dataset = load_dataset('roneneldan/TinyStories')
    
    # Create and train tokenizer
    print("Creating and training tokenizer...")
    tokenizer = create_and_train_tokenizer(dataset)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Tokenize all texts
    print("Tokenizing texts...")
    all_texts = dataset['train']['text'][:config.num_train_samples]
    all_tokens = []
    for text in all_texts:
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.token_to_id("<|endoftext|>"))  # Add EOS token
    
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    
    print(f"Total tokens: {len(tokens)}")
    print(f"GPU Memory after tokenization: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB")
    
    # Create train/val split
    n = int(0.9 * len(tokens))
    train_data = TextDataset(tokens[:n], config.block_size, device)
    val_data = TextDataset(tokens[n:], config.block_size, device)

    print("Creating dataloaders...")
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.micro_batch_size,
        shuffle=True,
        pin_memory=False,  # Already on GPU
        num_workers=0,  # No need for workers as data is on GPU
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=config.micro_batch_size,
        shuffle=False,
        pin_memory=False,  # Already on GPU
        num_workers=0,  # No need for workers as data is on GPU
        drop_last=True
    )

    print("Initializing model...")
    # Initialize model
    from diff_transformer import DiffTransformer
    '''model = DiffTransformer(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout
    ).to(device) '''
    '''model = AlternatingDiffTransformer(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
        n_terms=config.n_terms
    ).to(device)
    '''
    model = StandardTransformer(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head * 2,  # Double the heads since each head is smaller
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout
    ).to(device)

    print(f"GPU Memory after model init: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB")

    # Initialize optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_iters,
        max_steps=config.max_iters,
        min_lr=config.min_lr
    )

    # Enable autocast for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    iter_num = 0
    grad_acc_counter = 0

    model.train()
    while iter_num < config.max_iters:
        for batch_idx, (X, Y) in enumerate(train_loader):
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
                loss = loss / config.grad_acc_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            grad_acc_counter += 1
            
            # Gradient accumulation
            if grad_acc_counter == config.grad_acc_steps:
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scheduler.step()
                grad_acc_counter = 0
                iter_num += 1
                
                # Logging
                if iter_num % config.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"iter {iter_num}: loss {loss.item() * config.grad_acc_steps:.4f}, lr {lr:.2e}")
                    wandb.log({
                        'iter': iter_num,
                        'loss': loss.item() * config.grad_acc_steps,
                        'learning_rate': lr,
                        'gpu_memory': torch.cuda.memory_allocated(device)/1024**2
                    })
                
                # Evaluation
                if iter_num % config.eval_interval == 0:
                    losses = estimate_loss(model, train_loader, val_loader, config)
                    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    wandb.log({
                        'iter': iter_num,
                        'train_loss': losses['train'],
                        'val_loss': losses['val']
                    })
                    
                    # Save best model
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        print(f"Saving best model with val loss: {best_val_loss:.4f}")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config
                        }, 'best_model.pt')
            
            if iter_num >= config.max_iters:
                break

        # Print memory stats at the end of each epoch
        print(f"GPU Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB")

    wandb.finish()

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.backends.cudnn.deterministic = True
    
    # Initialize config
    config = TrainingConfig()
    
    # Start training
    train(config)