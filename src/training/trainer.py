import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
import logging
import os
from tqdm import tqdm
import wandb
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class FewTopNERTrainer:
    """Enhanced trainer for multilingual FewTopNER"""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        train_dataloaders: Dict[str, DataLoader],
        val_dataloaders: Dict[str, DataLoader],
        test_dataloaders: Optional[Dict[str, DataLoader]] = None,
        episode_loader = None
    ):
        self.model = model
        self.config = config
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.test_dataloaders = test_dataloaders
        self.episode_loader = episode_loader
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer with layer-wise learning rates
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {
            'entity_f1': float('-inf'),
            'topic_accuracy': float('-inf'),
            'combined_score': float('-inf')
        }
        
    def _create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'lr': float(self.config.optimizer.encoder_lr),
                'weight_decay': float(self.config.optimizer.weight_decay)
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'lr': float(self.config.optimizer.task_lr),
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            eps=float(self.config.optimizer.adam_epsilon),
            betas=(float(self.config.optimizer.beta1), 
                  float(self.config.optimizer.beta2))
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        total_steps = (
            self.config.training.num_epochs * 
            self.config.training.episodes_per_epoch * 
            len(self.train_dataloaders)
        )
        
        warmup_steps = int(total_steps * self.config.optimizer.warmup_ratio)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        """Main training loop with episode-based few-shot learning"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Log metrics
            self._log_metrics(
                {'train': train_metrics},
                {'validation': val_metrics}
            )
            
            # Save checkpoint if improved
            if self._is_improved(val_metrics):
                self._save_checkpoint()
        
        # Final test evaluation
        if self.test_dataloaders:
            test_metrics = self._test()
            self._log_metrics({'test': test_metrics})

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with few-shot episodes"""
        self.model.train()
        epoch_stats = {
            'loss': 0.0,
            'entity_loss': 0.0,
            'topic_loss': 0.0,
            'bridge_loss': 0.0,
            'contrastive_loss': 0.0
        }
        
        num_batches = 0
        progress_bar = tqdm(
            self.episode_loader,
            desc=f"Training epoch {self.current_epoch + 1}"
        )
        
        for batch, support_loader, query_loader in progress_bar:
            # Process support set
            support_loss = 0
            for support_batch in support_loader:
                support_batch = self._to_device(support_batch)
                support_outputs = self.model(
                    **support_batch,
                    support_set=None
                )
                support_loss += support_outputs['losses']['total']
            
            # Process query set
            query_loss = 0
            for query_batch in query_loader:
                query_batch = self._to_device(query_batch)
                query_outputs = self.model(
                    **query_batch,
                    support_set=support_batch
                )
                query_loss += query_outputs['losses']['total']
            
            # Compute total loss
            total_loss = (support_loss / len(support_loader) + 
                         query_loss / len(query_loader)) / 2
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.optimizer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            epoch_stats['loss'] += total_loss.item()
            epoch_stats['entity_loss'] += (
                query_outputs['losses']['entity'].item()
            )
            epoch_stats['topic_loss'] += (
                query_outputs['losses']['topic'].item()
            )
            epoch_stats['bridge_loss'] += (
                query_outputs['losses']['bridge'].item()
            )
            epoch_stats['contrastive_loss'] += (
                query_outputs['losses']['contrastive'].item()
            )
            
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        # Compute averages
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return epoch_stats

    def _validate(self) -> Dict[str, float]:
        """Validation across all languages"""
        self.model.eval()
        metrics = {lang: {} for lang in self.val_dataloaders.keys()}
        
        with torch.no_grad():
            for lang, dataloader in self.val_dataloaders.items():
                lang_metrics = self._evaluate_language(
                    dataloader,
                    lang,
                    "Validation"
                )
                metrics[lang] = lang_metrics
        
        # Compute average metrics across languages
        avg_metrics = self._average_metrics(metrics)
        return {**metrics, 'average': avg_metrics}

    def _evaluate_language(
        self,
        dataloader: DataLoader,
        language: str,
        phase: str
    ) -> Dict[str, float]:
        """Evaluate model on a specific language"""
        metrics = {
            'entity_f1': 0.0,
            'topic_accuracy': 0.0,
            'loss': 0.0
        }
        
        num_batches = 0
        for batch in tqdm(dataloader, desc=f"{phase} - {language}"):
            batch = self._to_device(batch)
            outputs = self.model(**batch)
            
            # Update metrics
            metrics['entity_f1'] += self._compute_f1(
                outputs['entity_outputs']['predictions'],
                batch['entity_labels']
            )
            metrics['topic_accuracy'] += self._compute_accuracy(
                outputs['topic_outputs']['predictions'],
                batch['topic_labels']
            )
            if 'losses' in outputs:
                metrics['loss'] += outputs['losses']['total'].item()
            
            num_batches += 1
        
        # Compute averages
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics

    def _is_improved(self, metrics: Dict) -> bool:
        """Check if model performance improved"""
        avg_metrics = metrics['average']
        combined_score = (
            avg_metrics['entity_f1'] + 
            avg_metrics['topic_accuracy']
        ) / 2
        
        if combined_score > self.best_metrics['combined_score']:
            self.best_metrics.update({
                'entity_f1': avg_metrics['entity_f1'],
                'topic_accuracy': avg_metrics['topic_accuracy'],
                'combined_score': combined_score
            })
            return True
        return False

    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        save_path = Path(self.config.output_dir) / 'checkpoints'
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_path / f'checkpoint-epoch{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if self._is_improved(self.best_metrics):
            best_path = save_path / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model: {best_path}")

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_metrics(self, *metric_dicts):
        """Log metrics to console and wandb"""
        for metrics in metric_dicts:
            for category, values in metrics.items():
                logger.info(f"\n{category.upper()} METRICS:")
                for lang, lang_metrics in values.items():
                    logger.info(f"\n{lang}:")
                    for name, value in lang_metrics.items():
                        logger.info(f"{name}: {value:.4f}")
                        
                        if wandb.run is not None:
                            wandb.log({
                                f"{category}/{lang}/{name}": value,
                                'epoch': self.current_epoch,
                                'global_step': self.global_step
                            })