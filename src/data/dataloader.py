import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Iterator, Optional
import logging
from .dataset import FewTopNERDataset, FewShotEpisode

class FewTopNERDataLoader:
    """
    DataLoader class that handles both regular batches and few-shot episodes
    for the FewTopNER model.
    """
    def __init__(
        self,
        ner_processor,
        topic_processor,
        config
    ):
        self.ner_processor = ner_processor
        self.topic_processor = topic_processor
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_language_data(
        self, 
        language: str,
        split: str = 'train'
    ) -> FewTopNERDataset:
        """
        Load data for a specific language and split
        
        Args:
            language: Language code
            split: Data split ('train', 'dev', 'test')
        Returns:
            FewTopNERDataset containing aligned NER and Topic data
        """
        # Load NER data
        ner_features = self.ner_processor.process_language(language)[split]
        
        # Load Topic data
        topic_features = self.topic_processor.process_language(language)[split]
        
        # Create combined dataset
        dataset = FewTopNERDataset(
            ner_features=ner_features,
            topic_features=topic_features,
            language=language
        )
        
        self.logger.info(f"Loaded {len(dataset)} examples for {language} {split}")
        return dataset

    def create_few_shot_episodes(
        self,
        dataset: FewTopNERDataset,
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int
    ) -> List[FewShotEpisode]:
        """
        Create few-shot episodes from a dataset
        
        Args:
            dataset: Source dataset
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to create
        Returns:
            List of FewShotEpisode objects
        """
        episodes = []
        
        for _ in range(n_episodes):
            # Get support and query sets for NER
            ner_support, ner_query = self.ner_processor.create_few_shot_episode(
                dataset.ner_features,
                n_way=n_way,
                k_shot=k_shot,
                n_query=n_query
            )
            
            # Get aligned topic features
            topic_support, topic_query = self.topic_processor.align_with_ner(
                dataset.topic_features,
                ner_support,
                ner_query
            )
            
            episode = FewShotEpisode(
                support_ner=ner_support,
                support_topic=topic_support,
                query_ner=ner_query,
                query_topic=topic_query,
                language=dataset.language
            )
            
            episodes.append(episode)
            
        return episodes

    def get_dataloader(
        self,
        dataset: FewTopNERDataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader for a dataset
        
        Args:
            dataset: FewTopNERDataset
            batch_size: Batch size (uses config if None)
            shuffle: Whether to shuffle the data
        Returns:
            PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle both NER and Topic features
        
        Args:
            batch: List of dictionaries containing features
        Returns:
            Dictionary of batched features
        """
        return {
            # NER features
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'label_ids': torch.stack([x['label_ids'] for x in batch]),
            
            # Topic features
            'topic_vector': torch.stack([x['topic_vector'] for x in batch]),
            'topic_label': torch.stack([x['topic_label'] for x in batch]),
            
            # Metadata
            'language': [x['language'] for x in batch]
        }

class FewShotEpisodeLoader:
    """
    Specialized loader for few-shot episodes
    """
    def __init__(
        self,
        episodes: List[FewShotEpisode],
        config
    ):
        self.episodes = episodes
        self.config = config
        
    def __len__(self) -> int:
        return len(self.episodes)
        
    def __iter__(self) -> Iterator[Tuple[DataLoader, DataLoader]]:
        """
        Iterate over episodes, returning support and query dataloaders
        """
        for episode in self.episodes:
            support_loader = DataLoader(
                episode.get_support(),
                batch_size=self.config.support_batch_size,
                shuffle=True
            )
            
            query_loader = DataLoader(
                episode.get_query(),
                batch_size=self.config.query_batch_size,
                shuffle=False
            )
            
            yield support_loader, query_loader