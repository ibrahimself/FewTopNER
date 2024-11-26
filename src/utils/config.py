from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os
import json
import yaml

@dataclass
class ModelConfig:
    """Base configuration for model architecture"""
    # Model parameters
    model_name: str = "xlm-roberta-base"
    hidden_size: int = 768  # XLM-RoBERTa hidden size
    dropout: float = 0.1
    max_length: int = 512
    
    # Feature dimensions
    projection_size: int = 512
    entity_feature_size: int = 256
    topic_feature_size: int = 256
    prototype_dim: int = 128
    contrastive_dim: int = 128
    
    # Task-specific parameters
    num_entity_labels: int = 9  # Including 'O' tag
    num_topics: int = 100
    
    # Transformer settings
    num_attention_heads: int = 8
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    attention_dropout: float = 0.1
    
    # Language settings
    languages: List[str] = field(default_factory=lambda: ['en', 'fr', 'de', 'es', 'it'])
    language_adapters: bool = True
    
    # Cross-lingual settings
    use_language_embeddings: bool = True
    language_embedding_dim: int = 32
    cross_lingual_sharing: bool = True
    
    # Other architecture settings
    use_crf: bool = True
    use_language_adapters: bool = True
    shared_encoder_layers: int = 12
    task_specific_layers: int = 2

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration"""
    wikineural_path: str = "data/ner/wikineural"
    wikipedia_base_path: str = "data/topic/wikipedia"  # Added this
    wiki_dump_date: str = "20231101"
    max_ner_length: int = 128
    min_text_length: int = 100
    max_text_length: int = 1000
    min_word_freq: int = 5
    max_word_freq: float = 0.7
    max_vocab_size: int = 50000
    cache_dir: str = "cache"
    use_cache: bool = True

@dataclass
class DataConfig:
    """Configuration for datasets and dataloaders"""
    # Paths and splits
    data_dir: str = "data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    # Batch sizes
    train_batch_size: int = 32
    eval_batch_size: int = 64
    support_batch_size: int = 16
    query_batch_size: int = 16
    
    # Few-shot episode settings
    n_way: int = 5
    k_shot: int = 5
    n_query: int = 10
    min_examples_per_class: int = 20
    num_languages_per_episode: int = 3
    
    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class OptimizerConfig:
    """Configuration for optimization"""
    # Learning rates
    encoder_lr: float = 2e-5
    task_lr: float = 1e-4
    bridge_lr: float = 1e-4
    
    # Scheduler
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # AdamW parameters
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Training schedule
    num_epochs: int = 10
    episodes_per_epoch: int = 100
    val_episodes: int = 50
    test_episodes: int = 100
    
    # Loss weights
    entity_loss_weight: float = 1.0
    topic_loss_weight: float = 1.0
    bridge_loss_weight: float = 0.1
    contrastive_loss_weight: float = 0.1
    prototype_loss_weight: float = 0.1
    
    # Device settings
    device: str = "cuda"
    fp16: bool = False
    
    # Checkpointing
    output_dir: str = "outputs"
    save_steps: int = 1000
    save_total_limit: int = 5
    
    # Logging
    logging_steps: int = 100
    evaluation_strategy: str = "epoch"
    log_level: str = "info"

    use_wandb: bool = True
    wandb_project: str = "FewTopNER"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None

@dataclass
class BridgeConfig:
    """Configuration for cross-task bridge module"""
    # Attention settings
    bridge_num_heads: int = 8
    attention_dropout: float = 0.1
    temperature: float = 0.07
    
    # Feature fusion
    use_language_adapters: bool = True
    use_task_gates: bool = True
    gate_dropout: float = 0.1
    
    # Loss weights
    alignment_weight: float = 0.1
    consistency_weight: float = 0.1

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    entity_aug_prob: float = 0.3
    context_aug_prob: float = 0.3
    translation_aug_prob: float = 0.2
    mixing_aug_prob: float = 0.2
    entity_sub_prob: float = 0.5
    synonym_aug_prob: float = 0.3
    word_order_aug_prob: float = 0.2
    topic_aug_prob: float = 0.3
    example_texts: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    # NER metrics
    use_seqeval: bool = True
    entity_scheme: str = "BIO2"
    
    # Topic metrics
    compute_coherence: bool = True
    compute_diversity: bool = True
    
    # Cross-lingual evaluation
    evaluate_transfer: bool = True
    source_languages: List[str] = field(default_factory=lambda: ['en'])
    
    # Few-shot evaluation
    eval_episodes: int = 1000
    max_eval_samples: Optional[int] = None

class FewTopNERConfig:
    """Complete configuration for FewTopNER"""
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        preprocessing_config: Optional[Dict] = None,
        data_config: Optional[Dict] = None,
        optimizer_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        bridge_config: Optional[Dict] = None,
        evaluation_config: Optional[Dict] = None
    ):
        self.model = ModelConfig(**(model_config or {}))
        self.preprocessing = PreprocessingConfig(**(preprocessing_config or {}))
        self.data = DataConfig(**(data_config or {}))
        self.optimizer = OptimizerConfig(**(optimizer_config or {}))
        self.training = TrainingConfig(**(training_config or {}))
        self.bridge = BridgeConfig(**(bridge_config or {}))
        self.evaluation = EvaluationConfig(**(evaluation_config or {}))
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'FewTopNERConfig':
        return cls(
            model_config=config_dict.get('model', {}),
            preprocessing_config=config_dict.get('preprocessing', {}),
            data_config=config_dict.get('data', {}),
            optimizer_config=config_dict.get('optimizer', {}),
            training_config=config_dict.get('training', {}),
            bridge_config=config_dict.get('bridge', {}),
            evaluation_config=config_dict.get('evaluation', {})
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'FewTopNERConfig':
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif file_path.endswith('.yaml'):
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml")
        return cls.from_dict(config_dict)
    
    def save(self, file_path: str):
        config_dict = {
            'model': self.model.to_dict(),
            'preprocessing': self.preprocessing.__dict__,
            'data': self.data.__dict__,
            'optimizer': self.optimizer.__dict__,
            'training': self.training.__dict__,
            'bridge': self.bridge.__dict__,
            'evaluation': self.evaluation.__dict__
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_path.endswith('.json'):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        elif file_path.endswith('.yaml'):
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f)
        else:
            raise ValueError("Config file must be .json or .yaml")
