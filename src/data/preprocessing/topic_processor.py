# src/data/preprocessing/topic_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from transformers import XLMRobertaTokenizer

class CulturaXTopicProcessor:
    def __init__(self, config):
        """
        Initialize the topic processor for CulturaX data.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.languages = {'fr': 'french', 'en': 'english', 
                         'es': 'spanish', 'it': 'italian', 'de': 'german'}
        
        # Initialize tokenizers and language models
        self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.language_models = {}
        self.stopwords = {}
        
        # Load language-specific resources
        for lang in self.languages.keys():
            try:
                self.language_models[lang] = spacy.load(f'{lang}_core_news_sm')
                self.stopwords[lang] = set(stopwords.words(self.languages[lang]))
            except:
                print(f"Warning: Could not load language model for {lang}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load parquet file and perform initial filtering.
        
        Args:
            file_path: Path to the parquet file
        Returns:
            Filtered DataFrame
        """
        df = pd.read_parquet(file_path)
        
        # Remove empty or very short texts
        df = df[df['text'].notna() & (df['text'].str.len() > self.config.min_text_length)]
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df

    def clean_text(self, text: str, language: str) -> str:
        """
        Clean text for topic modeling.
        
        Args:
            text: Input text
            language: Language code
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def tokenize_and_lemmatize(self, text: str, language: str) -> List[str]:
        """
        Tokenize and lemmatize text using language-specific models.
        
        Args:
            text: Input text
            language: Language code
        Returns:
            List of processed tokens
        """
        if language in self.language_models:
            doc = self.language_models[language](text)
            tokens = [token.lemma_ for token in doc 
                     if token.lemma_ not in self.stopwords.get(language, set())
                     and not token.is_punct and not token.is_space]
        else:
            # Fallback to basic tokenization
            tokens = word_tokenize(text)
            tokens = [token for token in tokens 
                     if token not in self.stopwords.get(language, set())]
        
        return tokens

    def process_document(self, doc: pd.Series, language: str) -> Dict:
        """
        Process a single document.
        
        Args:
            doc: Document series from DataFrame
            language: Language code
        Returns:
            Processed document dictionary
        """
        cleaned_text = self.clean_text(doc['text'], language)
        tokens = self.tokenize_and_lemmatize(cleaned_text, language)
        
        return {
            'original_text': doc['text'],
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'source': doc['source'],
            'timestamp': doc['timestamp'],
            'url': doc['url']
        }

    def process_language_data(self, df: pd.DataFrame, language: str) -> List[Dict]:
        """
        Process all documents for a given language.
        
        Args:
            df: DataFrame containing documents
            language: Language code
        Returns:
            List of processed documents
        """
        processed_docs = []
        
        for _, row in df.iterrows():
            try:
                processed_doc = self.process_document(row, language)
                if len(processed_doc['tokens']) >= self.config.min_tokens:
                    processed_docs.append(processed_doc)
            except Exception as e:
                print(f"Error processing document: {e}")
                continue
                
        return processed_docs

    def filter_quality_docs(self, docs: List[Dict]) -> List[Dict]:
        """
        Filter documents based on quality metrics.
        
        Args:
            docs: List of processed documents
        Returns:
            Filtered list of documents
        """
        filtered_docs = []
        
        for doc in docs:
            # Calculate quality metrics
            token_length = len(doc['tokens'])
            unique_tokens = len(set(doc['tokens']))
            token_diversity = unique_tokens / token_length if token_length > 0 else 0
            
            # Apply quality filters
            if (token_length >= self.config.min_tokens and 
                token_length <= self.config.max_tokens and
                token_diversity >= self.config.min_token_diversity):
                filtered_docs.append(doc)
                
        return filtered_docs

    def process_file(self, file_path: str, language: str) -> List[Dict]:
        """
        Process a single parquet file.
        
        Args:
            file_path: Path to parquet file
            language: Language code
        Returns:
            List of processed and filtered documents
        """
        # Load data
        df = self.load_data(file_path)
        
        # Process documents
        processed_docs = self.process_language_data(df, language)
        
        # Filter for quality
        filtered_docs = self.filter_quality_docs(processed_docs)
        
        return filtered_docs