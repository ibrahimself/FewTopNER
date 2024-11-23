import os
import torch
import logging
import argparse
import wandb
from datetime import datetime
from typing import Dict, Optional

from src.model.fewtopner import FewTopNER
from src.data.dataset import FewTopNERDataset
from src.data.preprocessing.ner_processor import NERProcessor
from src.data.preprocessing.topic_processor import TopicProcessor
from src.data.dataloader import FewTopNERDataLoader
from src.training.trainer import FewTopNERTrainer
from src.utils.config import FewTopNERConfig
from src.utils.multilingual import LanguageManager

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = os.path.join(config.training.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'fewtopner_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def setup_wandb(config, project_name: str = "FewTopNER"):
    """Initialize Weights & Biases"""
    wandb.init(
        project=project_name,
        config=config.__dict__,
        name=f"FewTopNER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

def load_data(config) -> Dict[str, FewTopNERDataset]:
    """
    Load and preprocess data for all languages
    
    Args:
        config: FewTopNERConfig object
    Returns:
        Dictionary of datasets for each split
    """
    logging.info("Loading and preprocessing data...")
    
    # Initialize processors
    ner_processor = NERProcessor(config)
    topic_processor = TopicProcessor(config)
    
    datasets = {}
    for split in ['train', 'dev', 'test']:
        # Process NER data
        ner_data = {}
        for lang in config.data.languages:
            lang_data = ner_processor.process_language(
                language=lang,
                split=split
            )
            if lang_data is not None:
                ner_data[lang] = lang_data
                
        # Process Topic data
        topic_data = {}
        for lang in config.data.languages:
            lang_data = topic_processor.process_language(
                language=lang,
                split=split
            )
            if lang_data is not None:
                topic_data[lang] = lang_data
                
        # Create combined dataset
        datasets[split] = FewTopNERDataset(
            ner_data=ner_data,
            topic_data=topic_data,
            config=config
        )
        
        logging.info(f"Loaded {len(datasets[split])} examples for {split}")
        
    return datasets

def train(config, datasets: Dict[str, FewTopNERDataset]):
    """
    Train FewTopNER model
    
    Args:
        config: FewTopNERConfig object
        datasets: Dictionary of datasets
    """
    logging.info("Initializing model and trainer...")
    
    # Initialize model
    model = FewTopNER(config)
    
    # Initialize trainer
    trainer = FewTopNERTrainer(
        model=model,
        config=config,
        train_dataset=datasets['train'],
        val_dataset=datasets['dev'],
        test_dataset=datasets['test']
    )
    
    # Start training
    logging.info("Starting training...")
    trainer.train()
    
    # Final evaluation
    logging.info("Running final evaluation...")
    test_metrics = trainer.evaluate(datasets['test'])
    
    return test_metrics

def evaluate(
    config,
    datasets: Dict[str, FewTopNERDataset],
    checkpoint_path: Optional[str] = None
):
    """
    Evaluate FewTopNER model
    
    Args:
        config: FewTopNERConfig object
        datasets: Dictionary of datasets
        checkpoint_path: Optional path to model checkpoint
    """
    logging.info("Initializing model for evaluation...")
    
    # Initialize model
    model = FewTopNER(config)
    
    # Load checkpoint if provided
    if checkpoint_path:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize trainer
    trainer = FewTopNERTrainer(
        model=model,
        config=config,
        train_dataset=None,
        val_dataset=None,
        test_dataset=datasets['test']
    )
    
    # Run evaluation
    logging.info("Running evaluation...")
    metrics = trainer.evaluate(datasets['test'])
    
    return metrics

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Run mode')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint for evaluation')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    args = parser.parse_args()
    
    # Load configuration
    if args.config.endswith('.json'):
        config = FewTopNERConfig.from_json(args.config)
    elif args.config.endswith('.yaml'):
        config = FewTopNERConfig.from_yaml(args.config)
    else:
        raise ValueError("Config file must be .json or .yaml")
    
    # Setup logging
    setup_logging(config)
    
    # Initialize wandb
    if not args.no_wandb:
        setup_wandb(config)
    
    # Load data
    datasets = load_data(config)
    
    if args.mode == 'train':
        # Train model
        test_metrics = train(config, datasets)
        logging.info("Final test metrics:")
        for name, value in test_metrics.items():
            logging.info(f"{name}: {value:.4f}")
            
    else:
        # Evaluate model
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for evaluation mode")
            
        metrics = evaluate(config, datasets, args.checkpoint)
        logging.info("Evaluation metrics:")
        for name, value in metrics.items():
            logging.info(f"{name}: {value:.4f}")
    
    # Close wandb
    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()