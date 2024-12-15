import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import v_measure_score, normalized_mutual_info_score
import logging
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class FewTopNERMetrics:
    """Enhanced metrics for joint NER and Topic evaluation"""
    
    def __init__(self, config):
        self.config = config
        
        # NER label mapping from WikiNEuRal
        self.ner_label_map = {
            0: 'O',
            1: 'B-PER', 2: 'I-PER',
            3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC',
            7: 'B-MISC', 8: 'I-MISC'
        }
        
        # Languages supported
        self.languages = ['en', 'fr', 'de', 'es', 'it']
        
        self.reset()
        
    def reset(self):
        """Reset metric accumulators"""
        # Per-language NER metrics
        self.ner_scores = {
            lang: {
                'predictions': [],
                'labels': [],
                'f1_scores': [],
                'precision_scores': [],
                'recall_scores': []
            }
            for lang in self.languages
        }
        
        # Per-language topic metrics
        self.topic_scores = {
            lang: {
                'predictions': [],
                'labels': [],
                'features': [],
                'coherence_scores': [],
                'nmi_scores': []
            }
            for lang in self.languages
        }
        
        # Cross-lingual metrics
        self.cross_lingual_scores = defaultdict(list)
        
        # Few-shot episode metrics
        self.episode_scores = defaultdict(list)

    def convert_to_bio_tags(
        self,
        label_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[str]]:
        """Convert label IDs to BIO tags"""
        predictions = []
        for seq_ids, mask in zip(label_ids, attention_mask):
            valid_ids = seq_ids[mask.bool()]
            tags = [self.ner_label_map[id.item()] for id in valid_ids if id != -100]
            predictions.append(tags)
        return predictions

    def update_ner_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor
    ):
        """Update NER metrics for each language"""
        # Convert tensors to BIO tags
        pred_tags = self.convert_to_bio_tags(predictions, attention_mask)
        true_tags = self.convert_to_bio_tags(labels, attention_mask)
        
        # Update per-language metrics
        for i, lang_id in enumerate(language_ids):
            lang = self.languages[lang_id.item()]
            lang_scores = self.ner_scores[lang]
            
            # Store predictions and labels
            lang_scores['predictions'].extend([pred_tags[i]])
            lang_scores['labels'].extend([true_tags[i]])
            
            # Compute scores
            try:
                f1 = f1_score([true_tags[i]], [pred_tags[i]])
                precision = precision_score([true_tags[i]], [pred_tags[i]])
                recall = recall_score([true_tags[i]], [pred_tags[i]])
                
                lang_scores['f1_scores'].append(f1)
                lang_scores['precision_scores'].append(precision)
                lang_scores['recall_scores'].append(recall)
            except:
                continue

    def update_topic_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        language_ids: torch.Tensor,
        texts: List[str]
    ):
        """Update topic modeling metrics for each language"""
        for i, lang_id in enumerate(language_ids):
            lang = self.languages[lang_id.item()]
            lang_scores = self.topic_scores[lang]
            
            # Store predictions, labels, and features
            lang_scores['predictions'].append(predictions[i].cpu())
            lang_scores['labels'].append(labels[i].cpu())
            lang_scores['features'].append(features[i].cpu())
            
            # Compute NMI score
            nmi = normalized_mutual_info_score(
                labels[i].cpu().numpy(),
                predictions[i].cpu().numpy()
            )
            lang_scores['nmi_scores'].append(nmi)
            
            # Compute topic coherence if texts are provided
            if texts:
                coherence = self.compute_topic_coherence(
                    texts[i],
                    predictions[i].cpu().numpy(),
                    features[i].cpu().numpy()
                )
                lang_scores['coherence_scores'].append(coherence)

    def compute_topic_coherence(
        self,
        text: str,
        prediction: int,
        features: np.ndarray
    ) -> float:
        """Compute topic coherence score"""
        # This is a simplified coherence calculation
        # In practice, you might want to use more sophisticated methods
        words = text.lower().split()
        word_pairs = [(words[i], words[i+1]) 
                     for i in range(len(words)-1)]
        
        coherence = 0
        if word_pairs:
            coherence = np.mean([
                1 - cosine(features, features)
                for w1, w2 in word_pairs
            ])
        return coherence

    def update_episode_metrics(
        self,
        support_ner_f1: float,
        support_topic_acc: float,
        query_ner_f1: float,
        query_topic_acc: float,
        languages: List[str]
    ):
        """Update few-shot episode metrics"""
        self.episode_scores['support_ner_f1'].append(support_ner_f1)
        self.episode_scores['support_topic_acc'].append(support_topic_acc)
        self.episode_scores['query_ner_f1'].append(query_ner_f1)
        self.episode_scores['query_topic_acc'].append(query_topic_acc)
        
        # Track cross-lingual performance
        lang_pair = '_'.join(sorted(languages))
        self.cross_lingual_scores[lang_pair].append({
            'ner_f1': query_ner_f1,
            'topic_acc': query_topic_acc
        })

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute all metrics"""
        metrics = {
            'per_language': {},
            'average': {},
            'cross_lingual': {},
            'few_shot': {}
        }
        
        # Compute per-language metrics
        for lang in self.languages:
            ner_metrics = self._compute_language_ner_metrics(lang)
            topic_metrics = self._compute_language_topic_metrics(lang)
            
            metrics['per_language'][lang] = {
                'ner': ner_metrics,
                'topic': topic_metrics
            }
        
        # Compute average metrics
        metrics['average'] = self._compute_average_metrics(
            metrics['per_language']
        )
        
        # Compute cross-lingual metrics
        metrics['cross_lingual'] = self._compute_cross_lingual_metrics()
        
        # Compute few-shot metrics
        metrics['few_shot'] = self._compute_few_shot_metrics()
        
        return metrics

    def _compute_language_ner_metrics(self, lang: str) -> Dict[str, float]:
        """Compute NER metrics for a specific language"""
        scores = self.ner_scores[lang]
        
        metrics = {
            'f1': np.mean(scores['f1_scores']) if scores['f1_scores'] else 0.0,
            'precision': np.mean(scores['precision_scores']) if scores['precision_scores'] else 0.0,
            'recall': np.mean(scores['recall_scores']) if scores['recall_scores'] else 0.0
        }
        
        # Add detailed classification report
        if scores['predictions'] and scores['labels']:
            report = classification_report(
                scores['labels'],
                scores['predictions'],
                output_dict=True
            )
            
            for entity_type, values in report.items():
                if isinstance(values, dict):
                    metrics[f'{entity_type}_f1'] = values['f1-score']
        
        return metrics

    def _compute_language_topic_metrics(self, lang: str) -> Dict[str, float]:
        """Compute topic metrics for a specific language"""
        scores = self.topic_scores[lang]
        
        metrics = {
            'nmi': np.mean(scores['nmi_scores']) if scores['nmi_scores'] else 0.0,
            'coherence': np.mean(scores['coherence_scores']) if scores['coherence_scores'] else 0.0
        }
        
        return metrics

    def _compute_average_metrics(
        self,
        language_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Compute average metrics across languages"""
        avg_metrics = defaultdict(list)
        
        for lang_metrics in language_metrics.values():
            for task, metrics in lang_metrics.items():
                for name, value in metrics.items():
                    avg_metrics[f'{task}_{name}'].append(value)
        
        return {
            name: np.mean(values)
            for name, values in avg_metrics.items()
        }

    def _compute_cross_lingual_metrics(self) -> Dict[str, float]:
        """Compute cross-lingual transfer metrics"""
        metrics = {}
        
        for lang_pair, scores in self.cross_lingual_scores.items():
            metrics[lang_pair] = {
                'ner_transfer': np.mean([s['ner_f1'] for s in scores]),
                'topic_transfer': np.mean([s['topic_acc'] for s in scores])
            }
        
        return metrics

    def _compute_few_shot_metrics(self) -> Dict[str, float]:
        """Compute few-shot learning metrics"""
        return {
            name: np.mean(scores) if scores else 0.0
            for name, scores in self.episode_scores.items()
        }