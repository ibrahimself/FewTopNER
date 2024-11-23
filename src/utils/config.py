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
    hidden_size: int = 768
    dropout: float = 0.1
    
    # Feature dimensions
    feature_projection_size: int = 512
    entity_feature_size: int = 256
    topic_feature_size: int = 256
    prototype_dim: int = 256
    
    # Task-specific parameters
    num_entity_labels: int = 9  # Including 'O' tag
    num_topics: int = 10
    o_label: int = 0  # ID for 'O' tag
    
    # Languages
    languages: List[str] = field(default_factory=lambda: ['en', 'fr', 'es', 'it', 'de'])
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class SharedEncoderConfig:
    """Configuration for shared encoder"""
    # Base parameters
    model_name: str = "xlm-roberta-base"
    hidden_size: int = 768
    projection_size: int = 512
    
    # Multilingual settings
    languages: List[str] = field(default_factory=lambda: ['en', 'fr', 'es', 'it', 'de'])
    language_specific_scaling: bool = True
    
    # Optimization
    dropout: float = 0.1
    gradient_scale: float = 1.0
    freeze_layers: int = 0

@dataclass
class EntityBranchConfig:
    """Configuration for entity recognition branch"""
    # LSTM parameters
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    
    # Feature dimensions
    entity_feature_size: int = 256
    prototype_dim: int = 256
    
    # Task parameters
    num_entity_labels: int = 9
    label_weights: Optional[List[float]] = None
    
    # Optimization
    dropout: float = 0.1

@dataclass
class TopicBranchConfig:
    """Configuration for topic modeling branch"""
    # Topic modeling parameters
    num_topics: int = 10
    topic_feature_size: int = 256
    prototype_dim: int = 256
    
    # LDA-BERT settings
    bert_dim: int = 768
    gamma: float = 45  # Scaling factor for LDA vectors
    
    # Optimization
    dropout: float = 0.1
    bert_batch_size: int = 32

@dataclass
class BridgeConfig:
    """Configuration for integration bridge"""
    # Feature dimensions
    hidden_size: int = 768
    entity_feature_size: int = 256
    topic_feature_size: int = 256
    
    # Attention parameters
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Integration parameters
    consistency_weight: float = 0.1
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Basic training parameters
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Batch sizes
    train_batch_size: int = 16
    eval_batch_size: int = 32
    
    # Learning rate schedule
    warmup_ratio: float = 0.1
    
    # Few-shot settings
    n_way: int = 5
    k_shot: int = 5
    n_query: int = 5
    episodes_per_epoch: int = 100
    val_episodes: int = 50
    test_episodes: int = 100
    min_examples_per_class: int = 10
    
    # Loss weights
    entity_loss_weight: float = 1.0
    topic_loss_weight: float = 1.0
    bridge_loss_weight: float = 0.1
    
    # Output
    output_dir: str = "outputs"
    logging_steps: int = 100
    save_steps: int = 1000
    
    # Evaluation
    evaluate_during_training: bool = True
    evaluation_strategy: str = "epoch"

@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Data paths
    data_path: str = "data"
    cache_dir: str = "cache"
    
    # Preprocessing
    max_seq_length: int = 128
    stride: int = 32
    truncation: bool = True
    
    # Topic modeling preprocessing
    min_doc_length: int = 50
    max_doc_length: int = 512
    min_word_freq: int = 5
    
    # Languages
    languages: List[str] = field(default_factory=lambda: ['en', 'fr', 'es', 'it', 'de'])
    
class FewTopNERConfig:
    """Main configuration class for FewTopNER"""
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        data_config: Optional[Dict] = None
    ):
        # Initialize sub-configs
        self.model = ModelConfig(**(model_config or {}))
        self.shared_encoder = SharedEncoderConfig()
        self.entity_branch = EntityBranchConfig()
        self.topic_branch = TopicBranchConfig()
        self.bridge = BridgeConfig()
        self.training = TrainingConfig(**(training_config or {}))
        self.data = DataConfig(**(data_config or {}))
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'FewTopNERConfig':
        """Create config from dictionary"""
        return cls(
            model_config=config_dict.get('model', {}),
            training_config=config_dict.get('training', {}),
            data_config=config_dict.get('data', {})
        )
        
    @classmethod
    def from_json(cls, json_path: str) -> 'FewTopNERConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FewTopNERConfig':
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
        
    def save(self, path: str):
        """Save config to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_dict = {
            'model': self.model.to_dict(),
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.endswith('.yaml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f)
        else:
            raise ValueError("Config file must be .json or .yaml")