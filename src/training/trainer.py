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

from .episode_builder import EpisodeBuilder
from .metrics import FewTopNERMetrics

class FewTopNERTrainer:
    """
    Trainer for FewTopNER model
    """
    def __init__(
        self,
        model: nn.Module,
        config,
        train_dataset,
        val_dataset,
        test_dataset=None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize episode builder and metrics
        self.episode_builder = EpisodeBuilder(config)
        self.metrics = FewTopNERMetrics(config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        
    def _create_optimizer(self):
        """Initialize optimizer with weight decay"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon
        )
        
    def _create_scheduler(self):
        """Initialize learning rate scheduler"""
        num_training_steps = self.config.num_epochs * len(self.train_dataset)
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint if improved
            if self._is_improved(val_metrics):
                self._save_checkpoint()
                
        # Final test evaluation
        if self.test_dataset is not None:
            self._test()
            
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.metrics.reset()
        
        # Create episodes for few-shot learning
        episodes = self.episode_builder.create_episodes(
            self.train_dataset,
            self.config.episodes_per_epoch
        )
        
        progress_bar = tqdm(episodes, desc="Training")
        for support_set, query_set in progress_bar:
            # Move data to device
            support_set = self._to_device(support_set)
            query_set = self._to_device(query_set)
            
            # Inner loop (support set)
            self.optimizer.zero_grad()
            support_outputs = self.model(
                **support_set,
                support_set=None  # No support set for support set training
            )
            support_loss = support_outputs['loss']
            support_loss.backward()
            
            # Outer loop (query set)
            query_outputs = self.model(
                **query_set,
                support_set=support_set
            )
            query_loss = query_outputs['loss']
            query_loss.backward()
            
            # Update model
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            self._update_metrics(
                support_outputs,
                query_outputs,
                support_set,
                query_set
            )
            
            # Update progress bar
            progress_bar.set_postfix({
                'support_loss': support_loss.item(),
                'query_loss': query_loss.item()
            })
            
            self.global_step += 1
            
        return self.metrics.get_all_metrics()
        
    def _validate(self) -> Dict[str, float]:
        """Validation phase"""
        self.model.eval()
        self.metrics.reset()
        
        # Create validation episodes
        episodes = self.episode_builder.create_episodes(
            self.val_dataset,
            self.config.val_episodes
        )
        
        with torch.no_grad():
            for support_set, query_set in tqdm(episodes, desc="Validation"):
                support_set = self._to_device(support_set)
                query_set = self._to_device(query_set)
                
                # Get predictions
                query_outputs = self.model(
                    **query_set,
                    support_set=support_set
                )
                
                # Update metrics
                self._update_metrics(
                    None,  # No support set metrics during validation
                    query_outputs,
                    None,
                    query_set
                )
                
        return self.metrics.get_all_metrics()
        
    def _test(self) -> Dict[str, float]:
        """Test phase"""
        self.logger.info("\nStarting test evaluation...")
        self.model.eval()
        self.metrics.reset()
        
        # Create test episodes
        episodes = self.episode_builder.create_episodes(
            self.test_dataset,
            self.config.test_episodes
        )
        
        with torch.no_grad():
            for support_set, query_set in tqdm(episodes, desc="Testing"):
                support_set = self._to_device(support_set)
                query_set = self._to_device(query_set)
                
                query_outputs = self.model(
                    **query_set,
                    support_set=support_set
                )
                
                self._update_metrics(
                    None,
                    query_outputs,
                    None,
                    query_set
                )
                
        # Log final test metrics
        test_metrics = self.metrics.get_all_metrics()
        self.logger.info("\nTest Results:")
        self._log_metrics({'test': test_metrics})
        
        return test_metrics
        
    def _update_metrics(
        self,
        support_outputs: Optional[Dict],
        query_outputs: Dict,
        support_set: Optional[Dict],
        query_set: Dict
    ):
        """Update metrics with batch outputs"""
        # Update NER metrics
        if query_outputs.get('entity_predictions') is not None:
            self.metrics.update_ner_metrics(
                predictions=query_outputs['entity_predictions'],
                targets=query_set['entity_labels'],
                language=query_set['language'][0]
            )
            
        # Update topic metrics
        if query_outputs.get('topic_predictions') is not None:
            self.metrics.update_topic_metrics(
                predictions=query_outputs['topic_predictions'],
                targets=query_set['topic_labels'],
                features=query_outputs['topic_features'],
                language=query_set['language'][0]
            )
            
        # Update episode metrics
        if support_outputs is not None:
            self.metrics.update_episode_metrics(
                support_accuracy=self._compute_accuracy(
                    support_outputs, support_set
                ),
                query_accuracy=self._compute_accuracy(
                    query_outputs, query_set
                ),
                adaptation_steps=1
            )
            
    def _compute_accuracy(
        self,
        outputs: Dict,
        batch: Dict
    ) -> float:
        """Compute accuracy for a batch"""
        entity_correct = (
            outputs['entity_predictions'] == 
            batch['entity_labels']
        ).float().mean()
        
        topic_correct = (
            outputs['topic_predictions'] == 
            batch['topic_labels']
        ).float().mean()
        
        return (entity_correct + topic_correct) / 2
        
    def _is_improved(self, metrics: Dict) -> bool:
        """Check if model improved"""
        current_metric = metrics['few_shot']['mean_query_accuracy']
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            return True
        return False
        
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f'checkpoint-{self.current_epoch}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def _log_metrics(self, *metric_dicts):
        """Log metrics to console and wandb"""
        for metrics in metric_dicts:
            for category, values in metrics.items():
                self.logger.info(f"\n{category} metrics:")
                for name, value in values.items():
                    self.logger.info(f"{name}: {value:.4f}")
                    
                    if wandb.run is not None:
                        wandb.log({
                            f"{category}/{name}": value,
                            'epoch': self.current_epoch,
                            'global_step': self.global_step
                        })
                        
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }