import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Iterator, Optional, Tuple
import logging
from pathlib import Path
from .dataset import FewTopNERDataset, FewShotEpisode, create_episodes
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FewTopNERDataLoader:
    """
    Enhanced DataLoader for FewTopNER that handles WikiNEuRal NER and Wikipedia Topic data
    """
    def __init__(
        self,
        wikineural_processor,
        wiki_topic_processor,
        config
    ):
        self.ner_processor = wikineural_processor
        self.topic_processor = wiki_topic_processor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache for processed data
        self.processed_data = {
            'ner': {},
            'topic': {},
            'combined': {}
        }
    
    def load_all_languages(self, splits: List[str] = ['train', 'dev', 'test']) -> Dict[str, Dict[str, FewTopNERDataset]]:
        """
        Load data for all supported languages
        
        Args:
            splits: List of data splits to load
        Returns:
            Dictionary of datasets by language and split
        """
        datasets = {}
        
        # Process NER data for all languages
        logger.info("Processing WikiNEuRal NER data...")
        ner_data = self.ner_processor.process_all_languages()
        
        # Load and process Wikipedia topic data
        logger.info("Processing Wikipedia topic data...")
        topic_data = self.topic_processor.process_all_languages()
        
        # Combine data for each language and split
        for lang in self.ner_processor.SUPPORTED_LANGUAGES:
            if lang not in ner_data or lang not in topic_data:
                logger.warning(f"Skipping {lang}: missing data")
                continue
                
            datasets[lang] = {}
            for split in splits:
                if split in ner_data[lang]:
                    dataset = self._create_combined_dataset(
                        ner_data[lang][split],
                        topic_data[lang],
                        lang,
                        split
                    )
                    datasets[lang][split] = dataset
                    logger.info(f"Created dataset for {lang} {split}: {len(dataset)} examples")
        
        return datasets
    
    def _create_combined_dataset(
        self,
        ner_data: Dict,
        topic_models: Tuple,
        language: str,
        split: str
    ) -> FewTopNERDataset:
        """Create combined dataset from NER and Topic data"""
        lda_model, dictionary = topic_models
        
        # Extract text from NER features
        texts = [
            ' '.join(feature.original_tokens)
            for feature in ner_data.features
            if hasattr(feature, 'original_tokens')
        ]
        
        # Create topic features
        topic_features = []
        for text in tqdm(texts, desc=f"Creating topic features for {language} {split}"):
            topic_vector = self.topic_processor.create_topic_features(
                text,
                language,
                lda_model,
                dictionary
            )
            if topic_vector is not None:
                topic_features.append({
                    'features': topic_vector.numpy(),
                    'language': language
                })
        
        return FewTopNERDataset(
            ner_features=ner_data.features,
            topic_features=topic_features,
            languages=language,
            is_few_shot=(split == 'train')
        )

    def create_few_shot_episodes(
        self,
        datasets: Dict[str, Dict[str, FewTopNERDataset]],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int
    ) -> List[FewShotEpisode]:
        """
        Create few-shot episodes from training data
        
        Args:
            datasets: Dictionary of datasets by language and split
            n_way: Number of entity types per episode
            k_shot: Number of support examples per entity type
            n_query: Number of query examples per entity type
            n_episodes: Number of episodes to create
        Returns:
            List of FewShotEpisode objects
        """
        # Prepare data for episode creation
        train_data = {
            lang: {
                'ner': datasets[lang]['train'].ner_features,
                'topic': datasets[lang]['train'].topic_features
            }
            for lang in datasets
            if 'train' in datasets[lang]
        }
        
        # Create episodes
        episodes = create_episodes(
            ner_data=train_data,
            topic_data=train_data,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes
        )
        
        return episodes

    def get_dataloader(
        self,
        dataset: FewTopNERDataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader for regular training/evaluation"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

class FewShotEpisodeLoader:
    """Enhanced loader for few-shot episodes"""
    def __init__(
        self,
        episodes: List[FewShotEpisode],
        config,
        infinite: bool = False
    ):
        self.episodes = episodes
        self.config = config
        self.infinite = infinite
        
    def __len__(self) -> int:
        return len(self.episodes)
        
    def __iter__(self) -> Iterator[Tuple[DataLoader, DataLoader]]:
        """Iterate over episodes, returning support and query dataloaders"""
        while True:
            for episode in self.episodes:
                # Get balanced batch from episode
                batch = episode.get_batch(
                    batch_size=self.config.episode_batch_size
                )
                
                # Create support loader
                support_loader = DataLoader(
                    episode.get_support(),
                    batch_size=self.config.support_batch_size,
                    shuffle=True,
                    collate_fn=episode.get_support().collate_fn,
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )
                
                # Create query loader
                query_loader = DataLoader(
                    episode.get_query(),
                    batch_size=self.config.query_batch_size,
                    shuffle=False,
                    collate_fn=episode.get_query().collate_fn,
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )
                
                yield batch, support_loader, query_loader
            
            if not self.infinite:
                break

def create_dataloaders(
    config,
    splits: List[str] = ['train', 'dev', 'test']
) -> Tuple[Dict[str, DataLoader], FewShotEpisodeLoader]:
    """
    Create all necessary dataloaders for training and evaluation
    
    Args:
        config: Configuration object
        splits: List of data splits to create loaders for
    Returns:
        Regular dataloaders and few-shot episode loader
    """
    # Initialize processors
    wikineural_processor = WikiNeuralProcessor(config)
    wiki_topic_processor = WikiTopicProcessor(config)
    
    # Create main dataloader
    loader = FewTopNERDataLoader(
        wikineural_processor,
        wiki_topic_processor,
        config
    )
    
    # Load all data
    datasets = loader.load_all_languages(splits)
    
    # Create regular dataloaders
    dataloaders = {
        split: {
            lang: loader.get_dataloader(datasets[lang][split])
            for lang in datasets
            if split in datasets[lang]
        }
        for split in splits
    }
    
    # Create few-shot episodes from training data
    episodes = loader.create_few_shot_episodes(
        datasets,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        n_episodes=config.n_episodes
    )
    
    # Create episode loader
    episode_loader = FewShotEpisodeLoader(
        episodes,
        config,
        infinite=True
    )
    
    return dataloaders, episode_loader