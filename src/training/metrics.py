import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
import logging

class FewTopNERMetrics:
    """
    Evaluation metrics for FewTopNER model
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric accumulators
        self.reset()
        
    def reset(self):
        """Reset all metric accumulators"""
        # NER metrics
        self.ner_predictions = []
        self.ner_targets = []
        
        # Topic metrics
        self.topic_predictions = []
        self.topic_targets = []
        self.topic_features = []
        
        # Few-shot metrics
        self.episode_scores = defaultdict(list)
        
        # Cross-lingual metrics
        self.language_scores = defaultdict(lambda: defaultdict(list))
        
    def update_ner_metrics(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        language: Optional[str] = None
    ):
        """
        Update NER metric accumulators
        
        Args:
            predictions: Predicted entity labels
            targets: True entity labels
            language: Optional language code
        """
        self.ner_predictions.extend(predictions)
        self.ner_targets.extend(targets)
        
        if language:
            # Compute F1 score for this batch
            batch_f1 = f1_score(predictions, targets)
            self.language_scores[language]['ner_f1'].append(batch_f1)
            
    def update_topic_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        language: Optional[str] = None
    ):
        """
        Update topic modeling metric accumulators
        
        Args:
            predictions: Predicted topic labels
            targets: True topic labels
            features: Topic features for clustering metrics
            language: Optional language code
        """
        self.topic_predictions.extend(predictions.cpu().numpy())
        self.topic_targets.extend(targets.cpu().numpy())
        self.topic_features.append(features.cpu().numpy())
        
        if language:
            # Compute V-measure score for this batch
            batch_v_score = v_measure_score(targets.cpu(), predictions.cpu())
            self.language_scores[language]['topic_v_score'].append(batch_v_score)
            
    def update_episode_metrics(
        self,
        support_accuracy: float,
        query_accuracy: float,
        adaptation_steps: int
    ):
        """
        Update few-shot learning metric accumulators
        
        Args:
            support_accuracy: Accuracy on support set
            query_accuracy: Accuracy on query set
            adaptation_steps: Number of adaptation steps
        """
        self.episode_scores['support_accuracy'].append(support_accuracy)
        self.episode_scores['query_accuracy'].append(query_accuracy)
        self.episode_scores['adaptation_steps'].append(adaptation_steps)
        
    def compute_ner_metrics(self) -> Dict[str, float]:
        """
        Compute final NER metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'precision': precision_score(self.ner_targets, self.ner_predictions),
            'recall': recall_score(self.ner_targets, self.ner_predictions),
            'f1': f1_score(self.ner_targets, self.ner_predictions)
        }
        
        # Detailed classification report
        report = classification_report(
            self.ner_targets,
            self.ner_predictions,
            output_dict=True
        )
        
        # Add per-entity type metrics
        for entity_type, scores in report.items():
            if isinstance(scores, dict):
                metrics[f'{entity_type}_f1'] = scores['f1-score']
                
        return metrics
        
    def compute_topic_metrics(self) -> Dict[str, float]:
        """
        Compute final topic modeling metrics
        
        Returns:
            Dictionary of metrics
        """
        # Convert lists to arrays
        predictions = np.array(self.topic_predictions)
        targets = np.array(self.topic_targets)
        features = np.concatenate(self.topic_features, axis=0)
        
        metrics = {
            'v_measure': v_measure_score(targets, predictions),
            'silhouette': silhouette_score(features, predictions),
            'davies_bouldin': davies_bouldin_score(features, predictions)
        }
        
        # Compute topic coherence if available
        if hasattr(self, 'compute_topic_coherence'):
            metrics['coherence'] = self.compute_topic_coherence(features, predictions)
            
        return metrics
        
    def compute_few_shot_metrics(self) -> Dict[str, float]:
        """
        Compute few-shot learning metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for metric_name, scores in self.episode_scores.items():
            metrics[f'mean_{metric_name}'] = np.mean(scores)
            metrics[f'std_{metric_name}'] = np.std(scores)
            
        # Compute adaptation rate
        support_accuracies = self.episode_scores['support_accuracy']
        adaptation_steps = self.episode_scores['adaptation_steps']
        
        if support_accuracies and adaptation_steps:
            metrics['adaptation_rate'] = np.mean([
                acc / steps
                for acc, steps in zip(support_accuracies, adaptation_steps)
            ])
            
        return metrics
        
    def compute_language_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-language metrics
        
        Returns:
            Nested dictionary of metrics per language
        """
        language_metrics = {}
        
        for language, scores in self.language_scores.items():
            language_metrics[language] = {
                metric: np.mean(values)
                for metric, values in scores.items()
            }
            
        return language_metrics
        
    def compute_cross_lingual_transfer(
        self,
        source_lang: str,
        target_lang: str
    ) -> float:
        """
        Compute cross-lingual transfer ratio
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        Returns:
            Transfer ratio score
        """
        source_f1 = np.mean(self.language_scores[source_lang]['ner_f1'])
        target_f1 = np.mean(self.language_scores[target_lang]['ner_f1'])
        
        return (target_f1 / source_f1) if source_f1 > 0 else 0.0
        
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute and combine all metrics
        
        Returns:
            Nested dictionary of all metrics
        """
        return {
            'ner': self.compute_ner_metrics(),
            'topic': self.compute_topic_metrics(),
            'few_shot': self.compute_few_shot_metrics(),
            'per_language': self.compute_language_metrics()
        }
        
    def log_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """
        Log metrics using logger
        
        Args:
            metrics: Dictionary of metrics to log
        """
        self.logger.info("=== Evaluation Metrics ===")
        
        for category, category_metrics in metrics.items():
            self.logger.info(f"\n{category.upper()} Metrics:")
            for name, value in category_metrics.items():
                self.logger.info(f"{name}: {value:.4f}")