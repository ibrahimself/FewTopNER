import dask.dataframe as dd
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
import logging
from transformers import XLMRobertaTokenizer
import spacy
from tqdm import tqdm
import numpy as np
import torch
from collections import defaultdict
import os
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikiTopicProcessor:
    """Process Wikipedia articles for topic modeling"""
    
    SUPPORTED_LANGUAGES = ['en', 'fr', 'de', 'es', 'it']
    WIKI_DUMP_DATE = "20231101"  # Wikipedia dump date
    
    SPACY_MODELS = {
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm',
        'de': 'de_core_news_sm',
        'es': 'es_core_news_sm',
        'it': 'it_core_news_sm'
    }
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        
        # Initialize spacy models for each language
        self.nlp_models = {}
        for lang in self.SUPPORTED_LANGUAGES:
            try:
                self.nlp_models[lang] = spacy.load(self.SPACY_MODELS[lang])
                logger.info(f"Loaded spaCy model for {lang}")
            except OSError:
                logger.warning(f"Could not load spaCy model for {lang}. Download it using:")
                logger.warning(f"python -m spacy download {self.SPACY_MODELS[lang]}")
        
        # Store DataFrames for each language
        self.wiki_dfs = {}
        
    def load_wikipedia_data(self) -> Dict[str, dd.DataFrame]:
        """Load Wikipedia parquet files for all languages"""
        for lang in self.SUPPORTED_LANGUAGES:
            try:
                # Construct path for each language
                wiki_path = os.path.join(
                    self.config.preprocessing.wikipedia_base_path,
                    f"{self.WIKI_DUMP_DATE}.{lang}",
                    "*.parquet"
                )
                
                logger.info(f"Loading Wikipedia data for {lang} from {wiki_path}")
                
                # Load parquet files using Dask
                self.wiki_dfs[lang] = dd.read_parquet(wiki_path)
                
                # Log basic information
                logger.info(f"Loaded {lang} Wikipedia data: {len(self.wiki_dfs[lang].columns)} columns")
                
            except Exception as e:
                logger.error(f"Error loading Wikipedia data for {lang}: {e}")
                continue
        
        return self.wiki_dfs

    def process_wiki_data(self, language: str) -> dd.DataFrame:
        """Process Wikipedia data for a specific language"""
        if language not in self.wiki_dfs:
            logger.error(f"No data loaded for language: {language}")
            return None
            
        df = self.wiki_dfs[language]
        
        try:
            # Basic filtering
            df = df[df.text.notnull()]
            
            # Clean text in parallel
            df['cleaned_text'] = df.text.map_partitions(
                lambda x: x.apply(lambda text: self._clean_text(text, language))
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {language} Wikipedia data: {e}")
            raise

    def _clean_text(self, text: str, language: str) -> str:
        """Clean Wikipedia text"""
        if not isinstance(text, str):
            return ""
            
        # Remove Wikipedia markup
        text = re.sub(r'\{\{[^\}]*\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # Clean wiki links
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)  # Remove references
        text = re.sub(r'==.*?==', '', text)  # Remove section headers
        
        # Remove URLs and special characters
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Basic cleaning
        text = text.lower().strip()
        text = ' '.join(text.split())
        
        return text

    def process_all_languages(self) -> Dict[str, Tuple[LdaModel, Dictionary]]:
        """Process Wikipedia data and train topic models for all languages"""
        # First load all Wikipedia data
        self.load_wikipedia_data()
        
        models = {}
        for lang in self.SUPPORTED_LANGUAGES:
            logger.info(f"Processing {lang} Wikipedia data...")
            
            # Process data for this language
            processed_df = self.process_wiki_data(lang)
            if processed_df is None:
                continue
            
            # Extract topics
            lda_model, dictionary = self.extract_topics(
                processed_df,
                lang,
                num_topics=self.config.model.num_topics
            )
            
            if lda_model is not None and dictionary is not None:
                models[lang] = (lda_model, dictionary)
                
                # Save models
                model_path = Path(self.config.preprocessing.model_dir) / 'topic_models' / lang
                model_path.mkdir(parents=True, exist_ok=True)
                lda_model.save(str(model_path / 'lda_model'))
                dictionary.save(str(model_path / 'dictionary'))
                
                logger.info(f"Saved topic model for {lang}")
        
        return models

    def create_few_shot_episodes(
        self,
        texts: Dict[str, List[str]],
        models: Dict[str, Tuple[LdaModel, Dictionary]],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int
    ) -> List[Dict]:
        """Create few-shot episodes for topic modeling"""
        episodes = []
        
        for _ in range(n_episodes):
            # Sample languages
            episode_langs = random.sample(
                self.SUPPORTED_LANGUAGES,
                min(len(self.SUPPORTED_LANGUAGES), 3)
            )
            
            support_set = []
            query_set = []
            
            for lang in episode_langs:
                if lang not in models or lang not in texts:
                    continue
                    
                lda_model, dictionary = models[lang]
                lang_texts = texts[lang]
                
                # Sample texts for this language
                sampled_texts = random.sample(
                    lang_texts,
                    min(len(lang_texts), k_shot + n_query)
                )
                
                # Create features
                for text in sampled_texts[:k_shot]:
                    features = self.create_topic_features(
                        text, lang, lda_model, dictionary
                    )
                    if features is not None:
                        support_set.append({
                            'features': features,
                            'language': lang
                        })
                
                for text in sampled_texts[k_shot:k_shot + n_query]:
                    features = self.create_topic_features(
                        text, lang, lda_model, dictionary
                    )
                    if features is not None:
                        query_set.append({
                            'features': features,
                            'language': lang
                        })
            
            if len(support_set) >= k_shot and len(query_set) >= n_query:
                episodes.append({
                    'support': support_set,
                    'query': query_set
                })
        
        return episodes
    
    def extract_topics(
        self,
        df: dd.DataFrame,
        language: str,
        num_topics: int = 100,
        chunksize: int = 2000
    ) -> Tuple[LdaModel, Dictionary]:
        """Extract topics using LDA
        
        Args:
            df: Dask DataFrame with Wikipedia articles
            language: Language code
            num_topics: Number of topics to extract
            chunksize: Size of chunks for processing
        
        Returns:
            LDA model and dictionary
        """
        try:
            nlp = self.nlp_models.get(language)
            if nlp is None:
                logger.error(f"No spaCy model available for {language}")
                return None, None

            # Process documents in chunks
            documents = []
            logger.info(f"Processing documents for {language}")
            
            for chunk in df.map_partitions(lambda x: x.text.tolist()).compute():
                chunk_docs = []
                for text in chunk:
                    if isinstance(text, str):
                        doc = nlp(text.lower())
                        # Extract lemmatized tokens, excluding stopwords and punctuation
                        tokens = [
                            token.lemma_ for token in doc 
                            if not token.is_stop and not token.is_punct 
                            and len(token.text) > 2
                        ]
                        if len(tokens) >= self.config.preprocessing.min_text_length:
                            chunk_docs.append(tokens)
                documents.extend(chunk_docs)

            if not documents:
                logger.error(f"No valid documents found for {language}")
                return None, None

            # Create dictionary
            logger.info(f"Creating dictionary for {language}")
            dictionary = Dictionary(documents)
            
            # Filter extremes
            dictionary.filter_extremes(
                no_below=self.config.preprocessing.min_word_freq,
                no_above=self.config.preprocessing.max_word_freq,
                keep_n=self.config.preprocessing.max_vocab_size
            )

            # Create corpus
            corpus = [dictionary.doc2bow(doc) for doc in documents]

            # Train LDA model
            logger.info(f"Training LDA model for {language}")
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=self.config.training.seed,
                alpha='auto',
                per_word_topics=True,
                chunksize=chunksize
            )

            return lda_model, dictionary

        except Exception as e:
            logger.error(f"Error extracting topics for {language}: {e}")
            return None, None
        
    def save_topic_model(
        self, 
        lda_model: LdaModel, 
        dictionary: Dictionary,
        language: str
    ):
        """Save topic model and dictionary"""
        try:
            model_dir = Path(self.config.preprocessing.cache_dir) / 'topic_models' / language
            model_dir.mkdir(parents=True, exist_ok=True)

            lda_model.save(str(model_dir / 'lda_model'))
            dictionary.save(str(model_dir / 'dictionary'))
            logger.info(f"Saved topic model for {language}")

        except Exception as e:
            logger.error(f"Error saving topic model for {language}: {e}")

    def load_topic_model(self, language: str) -> Tuple[Optional[LdaModel], Optional[Dictionary]]:
        """Load saved topic model and dictionary"""
        try:
            model_dir = Path(self.config.preprocessing.cache_dir) / 'topic_models' / language
            
            if not model_dir.exists():
                return None, None

            lda_path = model_dir / 'lda_model'
            dict_path = model_dir / 'dictionary'

            if lda_path.exists() and dict_path.exists():
                lda_model = LdaModel.load(str(lda_path))
                dictionary = Dictionary.load(str(dict_path))
                logger.info(f"Loaded topic model for {language}")
                return lda_model, dictionary

        except Exception as e:
            logger.error(f"Error loading topic model for {language}: {e}")
            
        return None, None