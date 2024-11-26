import os
import torch
import logging
import argparse
import wandb
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import yaml
import json
from tqdm import tqdm

from src.model.fewtopner import FewTopNER
from src.data.preprocessing.ner_processor import WikiNeuralProcessor
from src.data.preprocessing.topic_processor import WikiTopicProcessor
from src.data.dataloader import FewTopNERDataLoader
from src.training.trainer import FewTopNERTrainer
from src.training.episode_builder import EpisodeBuilder
from src.utils.config import FewTopNERConfig
from src.utils.multilingual import LanguageManager, CrossLingualAugmenter

logger = logging.getLogger(__name__)

def setup_environment(config):
    """Setup training environment"""
    # Create output directories
    output_dir = Path(config.training.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=getattr(logging, config.training.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'logs' / f'fewtopner_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    if config.training.use_wandb:
        wandb.init(
            project="FewTopNER",
            name=f"FewTopNER_{timestamp}",
            config=config.__dict__
        )
    
    return device

def setup_language_tools(config):
    """Setup language processing tools"""
    try:
        # Initialize language manager
        language_manager = LanguageManager(config)
        
        # Initialize augmenter
        augmenter = CrossLingualAugmenter(config, language_manager)
        
        return language_manager, augmenter
        
    except Exception as e:
        logger.error(f"Error setting up language tools: {e}")
        raise


def prepare_data(config, language_manager):
    """Prepare datasets and dataloaders"""
    logger.info("Preparing datasets...")
    
    # Initialize processors
    ner_processor = WikiNeuralProcessor(config)
    topic_processor = WikiTopicProcessor(config)
    
    # Process WikiNEuRal NER data
    logger.info("Processing WikiNEuRal data...")
    ner_datasets = ner_processor.process_all_languages()
    
    # Process Wikipedia topic data
    logger.info("Processing Wikipedia data...")
    topic_models = topic_processor.process_all_languages()
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloader = FewTopNERDataLoader(
        ner_processor=ner_processor,
        wiki_topic_processor=topic_processor,
        config=config
    )
    
    # Create episode builder
    episode_builder = EpisodeBuilder(config)
    
    # Create episode loader
    episode_loader = episode_builder.create_episode_loader(
        datasets=ner_datasets,
        num_episodes=config.training.episodes_per_epoch,
        infinite=True
    )
    
    return {
        'ner_datasets': ner_datasets,
        'topic_models': topic_models,
        'regular_loaders': dataloader,
        'episode_loader': episode_loader
    }

def build_model(config, device, topic_models):
    """Initialize FewTopNER model"""
    logger.info("Initializing model...")
    
    # Create model
    model = FewTopNER(config).to(device)
    
    # Set topic models
    model.topic_branch.set_lda_models(topic_models)
    
    return model

def train_model(
    model: FewTopNER,
    data_loaders: Dict,
    config: FewTopNERConfig,
    device: torch.device
):
    """Train FewTopNER model"""
    logger.info("Starting training...")
    
    # Initialize trainer
    trainer = FewTopNERTrainer(
        model=model,
        config=config,
        train_dataloaders=data_loaders['regular_loaders'],
        val_dataloaders=data_loaders['regular_loaders'],
        episode_loader=data_loaders['episode_loader']
    )
    
    # Train model
    trainer.train()
    
    return trainer

def evaluate_model(trainer: FewTopNERTrainer, test_loaders: Dict, device: torch.device):
    """Evaluate model on test set"""
    logger.info("Running evaluation...")
    
    metrics = {}
    
    # Evaluate NER performance
    ner_metrics = trainer.evaluate_ner(test_loaders)
    metrics['ner'] = ner_metrics
    
    # Evaluate topic modeling performance
    topic_metrics = trainer.evaluate_topics(test_loaders)
    metrics['topic'] = topic_metrics
    
    # Evaluate cross-lingual performance
    cross_lingual_metrics = trainer.evaluate_cross_lingual(test_loaders)
    metrics['cross_lingual'] = cross_lingual_metrics
    
    # Log metrics
    log_metrics(metrics)
    
    return metrics

def log_metrics(metrics: Dict):
    """Log evaluation metrics"""
    logger.info("\nEvaluation Results:")
    
    # Log NER metrics
    logger.info("\nNER Metrics:")
    for lang, scores in metrics['ner'].items():
        logger.info(f"\n{lang}:")
        logger.info(f"F1: {scores['f1']:.4f}")
        logger.info(f"Precision: {scores['precision']:.4f}")
        logger.info(f"Recall: {scores['recall']:.4f}")
    
    # Log topic metrics
    logger.info("\nTopic Modeling Metrics:")
    for lang, scores in metrics['topic'].items():
        logger.info(f"\n{lang}:")
        logger.info(f"Coherence: {scores['coherence']:.4f}")
        logger.info(f"NMI: {scores['nmi']:.4f}")
    
    # Log cross-lingual metrics
    logger.info("\nCross-lingual Transfer:")
    for pair, scores in metrics['cross_lingual'].items():
        logger.info(f"\n{pair}:")
        logger.info(f"NER Transfer: {scores['ner_transfer']:.4f}")
        logger.info(f"Topic Transfer: {scores['topic_transfer']:.4f}")
    
    if wandb.run:
        wandb.log(metrics)


def main():
    """Main execution pipeline"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='FewTopNER Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for evaluation')
    args = parser.parse_args()
    
    # Load configuration
    config = FewTopNERConfig.from_file(args.config)

    # Setup environment
    device = setup_environment(config)
    
    # Initialize language manager
    language_manager, augmenter = setup_language_tools(config)
    
    try:
        # Prepare data
        logger.info("Step 1: Preparing data...")
        data = prepare_data(config, language_manager)
        
        # Build model
        logger.info("Step 2: Building model...")
        model = build_model(config, device, data['topic_models'])
        
        if args.checkpoint:
            # Load checkpoint if provided
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        
        # Train model
        logger.info("Step 3: Training model...")
        trainer = train_model(model, data, config, device)
        
        # Evaluate model
        logger.info("Step 4: Evaluating model...")
        metrics = evaluate_model(trainer, data['regular_loaders'], device)
        
        # Save final results
        results_path = Path(config.training.output_dir) / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved results to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if 'language_manager' in locals():
            language_manager.cleanup()
        if 'augmenter' in locals():
            augmenter.cleanup()

if __name__ == '__main__':
    main()
