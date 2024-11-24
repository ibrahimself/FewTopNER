import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from copy import deepcopy
import spacy
from transformers import MarianMTTokenizer, MarianMTModel

class CrossLingualAugmenter:
    """Enhanced cross-lingual data augmentation for FewTopNER"""
    
    def __init__(self, config, language_manager):
        self.config = config
        self.lang_manager = language_manager
        
        # Load translation models for back-translation
        self.translation_models = self._load_translation_models()
        
        # Entity dictionaries for substitution
        self.entity_dictionaries = self._load_entity_dictionaries()
        
        # Cache for word translations
        self.translation_cache = defaultdict(dict)
        
        # Named entity sets per type and language
        self.entity_sets = self._initialize_entity_sets()
    
    def _load_translation_models(self) -> Dict[str, Tuple[MarianMTModel, MarianMTTokenizer]]:
        """Load translation models for each language pair"""
        models = {}
        language_pairs = [
            ('en', 'fr'), ('en', 'de'), ('en', 'es'), ('en', 'it'),
            ('fr', 'en'), ('de', 'en'), ('es', 'en'), ('it', 'en')
        ]
        
        for src, tgt in language_pairs:
            model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
            try:
                tokenizer = MarianMTTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                models[f'{src}-{tgt}'] = (model, tokenizer)
            except:
                continue
                
        return models

    def _initialize_entity_sets(self) -> Dict[str, Dict[str, Set[str]]]:
        """Initialize sets of entities for each type and language"""
        entity_sets = defaultdict(lambda: defaultdict(set))
        
        # Load entity lists for each language
        for lang in self.lang_manager.config.languages:
            nlp = self.lang_manager.nlp_models.get(lang)
            if nlp:
                # Add common entities from spaCy
                with nlp.select_pipes(enable=['ner']):
                    for text in self.entity_dictionaries.get(lang, []):
                        doc = nlp(text)
                        for ent in doc.ents:
                            entity_sets[lang][ent.label_].add(ent.text)
        
        return entity_sets

    def augment_ner_sample(
        self,
        tokens: List[str],
        labels: List[str],
        language: str
    ) -> List[Tuple[List[str], List[str]]]:
        """Apply multiple augmentation strategies for NER"""
        augmented_samples = []
        
        # Entity-based augmentations
        if random.random() < self.config.entity_aug_prob:
            # Entity substitution
            aug_tokens, aug_labels = self._substitute_entities(tokens, labels, language)
            augmented_samples.append((aug_tokens, aug_labels))
            
            # Entity switching
            aug_tokens, aug_labels = self._switch_entities(tokens, labels)
            augmented_samples.append((aug_tokens, aug_labels))
            
            # Entity insertion
            aug_tokens, aug_labels = self._insert_entities(tokens, labels, language)
            augmented_samples.append((aug_tokens, aug_labels))
        
        # Context-based augmentations
        if random.random() < self.config.context_aug_prob:
            # Context word substitution
            aug_tokens, aug_labels = self._substitute_context(tokens, labels, language)
            augmented_samples.append((aug_tokens, aug_labels))
            
            # Context word deletion
            aug_tokens, aug_labels = self._delete_context(tokens, labels)
            augmented_samples.append((aug_tokens, aug_labels))
        
        # Translation-based augmentations
        if random.random() < self.config.translation_aug_prob:
            # Back-translation
            aug_samples = self._back_translate(tokens, labels, language)
            augmented_samples.extend(aug_samples)
            
            # Pivot translation
            aug_samples = self._pivot_translate(tokens, labels, language)
            augmented_samples.extend(aug_samples)
        
        # Mixing-based augmentations
        if random.random() < self.config.mixing_aug_prob:
            # Entity mixing
            aug_tokens, aug_labels = self._mix_entities(tokens, labels, language)
            augmented_samples.append((aug_tokens, aug_labels))
            
            # Sentence mixing
            aug_tokens, aug_labels = self._mix_sentences(tokens, labels, language)
            augmented_samples.append((aug_tokens, aug_labels))
        
        return augmented_samples

    def augment_topic_sample(
        self,
        text: str,
        language: str
    ) -> List[str]:
        """Apply multiple augmentation strategies for topic modeling"""
        augmented_texts = []
        
        # Synonym replacement
        if random.random() < self.config.synonym_aug_prob:
            aug_text = self._replace_with_synonyms(text, language)
            augmented_texts.append(aug_text)
        
        # Back-translation
        if random.random() < self.config.translation_aug_prob:
            aug_texts = self._back_translate_text(text, language)
            augmented_texts.extend(aug_texts)
        
        # Word order perturbation
        if random.random() < self.config.word_order_aug_prob:
            aug_text = self._perturb_word_order(text, language)
            augmented_texts.append(aug_text)
        
        # Topic-specific augmentations
        if random.random() < self.config.topic_aug_prob:
            aug_texts = self._augment_topic_specific(text, language)
            augmented_texts.extend(aug_texts)
        
        return augmented_texts

    def _substitute_entities(
        self,
        tokens: List[str],
        labels: List[str],
        language: str
    ) -> Tuple[List[str], List[str]]:
        """Substitute entities with others of same type"""
        new_tokens = tokens.copy()
        new_labels = labels.copy()
        
        entity_spans = self._get_entity_spans(tokens, labels)
        for start, end, entity_type in entity_spans:
            if random.random() < self.config.entity_sub_prob:
                # Get replacement entity of same type
                replacement = self._get_random_entity(entity_type, language)
                if replacement:
                    replacement_tokens = replacement.split()
                    # Adjust tokens and labels
                    new_tokens[start:end] = replacement_tokens
                    new_labels[start:end] = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(replacement_tokens) - 1)
        
        return new_tokens, new_labels

    def _switch_entities(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Switch positions of two entities"""
        new_tokens = tokens.copy()
        new_labels = labels.copy()
        
        entity_spans = self._get_entity_spans(tokens, labels)
        if len(entity_spans) >= 2:
            span1, span2 = random.sample(entity_spans, 2)
            start1, end1, _ = span1
            start2, end2, _ = span2
            
            # Switch tokens
            tokens_1 = new_tokens[start1:end1]
            tokens_2 = new_tokens[start2:end2]
            new_tokens[start1:end1] = tokens_2
            new_tokens[start2:end2] = tokens_1
            
            # Switch labels
            labels_1 = new_labels[start1:end1]
            labels_2 = new_labels[start2:end2]
            new_labels[start1:end1] = labels_2
            new_labels[start2:end2] = labels_1
        
        return new_tokens, new_labels

    def _insert_entities(
        self,
        tokens: List[str],
        labels: List[str],
        language: str
    ) -> Tuple[List[str], List[str]]:
        """Insert new entities at random positions"""
        new_tokens = tokens.copy()
        new_labels = labels.copy()
        
        # Get valid insertion points (between non-entity tokens)
        insertion_points = []
        for i in range(len(labels) - 1):
            if labels[i] == 'O' and labels[i + 1] == 'O':
                insertion_points.append(i + 1)
        
        if insertion_points:
            # Insert random entity
            insert_pos = random.choice(insertion_points)
            entity_type = random.choice(['PER', 'ORG', 'LOC', 'MISC'])
            new_entity = self._get_random_entity(entity_type, language)
            
            if new_entity:
                entity_tokens = new_entity.split()
                entity_labels = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(entity_tokens) - 1)
                
                new_tokens[insert_pos:insert_pos] = entity_tokens
                new_labels[insert_pos:insert_pos] = entity_labels
        
        return new_tokens, new_labels

    def _back_translate(
        self,
        tokens: List[str],
        labels: List[str],
        language: str
    ) -> List[Tuple[List[str], List[str]]]:
        """Augment by translating to another language and back"""
        augmented_samples = []
        text = ' '.join(tokens)
        
        # Choose target languages for back-translation
        target_languages = [lang for lang in self.config.languages if lang != language]
        
        for target_lang in random.sample(target_languages, min(2, len(target_languages))):
            try:
                # Translate to target language
                forward_model, forward_tokenizer = self.translation_models[f'{language}-{target_lang}']
                translated = self._translate_text(text, forward_model, forward_tokenizer)
                
                # Translate back to source language
                backward_model, backward_tokenizer = self.translation_models[f'{target_lang}-{language}']
                back_translated = self._translate_text(translated, backward_model, backward_tokenizer)
                
                # Align labels with new tokens
                new_tokens = back_translated.split()
                new_labels = self._align_labels_after_translation(new_tokens, tokens, labels)
                
                if new_labels:
                    augmented_samples.append((new_tokens, new_labels))
                    
            except KeyError:
                continue
        
        return augmented_samples

    def _translate_text(
        self,
        text: str,
        model: MarianMTModel,
        tokenizer: MarianMTTokenizer
    ) -> str:
        """Translate text using MarianMT model"""
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _align_labels_after_translation(
        self,
        new_tokens: List[str],
        original_tokens: List[str],
        original_labels: List[str]
    ) -> Optional[List[str]]:
        """Align labels after translation using token similarity"""
        try:
            new_labels = ['O'] * len(new_tokens)
            entity_spans = self._get_entity_spans(original_tokens, original_labels)
            
            for start, end, entity_type in entity_spans:
                original_entity = ' '.join(original_tokens[start:end])
                # Find best match in new tokens
                best_match = self._find_best_match(original_entity, new_tokens)
                if best_match:
                    match_start, match_end = best_match
                    new_labels[match_start] = f"B-{entity_type}"
                    new_labels[match_start + 1:match_end] = [f"I-{entity_type}"] * (match_end - match_start - 1)
            
            return new_labels
            
        except Exception:
            return None

    def _augment_topic_specific(
        self,
        text: str,
        language: str
    ) -> List[str]:
        """Topic-specific augmentations"""
        augmented_texts = []
        nlp = self.lang_manager.nlp_models.get(language)
        
        if nlp:
            doc = nlp(text)
            
            # Extract key phrases
            key_phrases = []
            for chunk in doc.noun_chunks:
                if not chunk.root.is_stop:
                    key_phrases.append(chunk.text)
            
            if key_phrases:
                # Create variations with key phrase substitution
                for _ in range(2):
                    aug_text = text
                    phrase_to_replace = random.choice(key_phrases)
                    synonym = self._get_phrase_synonym(phrase_to_replace, language)
                    if synonym:
                        aug_text = aug_text.replace(phrase_to_replace, synonym)
                        augmented_texts.append(aug_text)
        
        return augmented_texts

    def _get_phrase_synonym(
        self,
        phrase: str,
        language: str
    ) -> Optional[str]:
        """Get synonym for a phrase in given language"""
        if language == 'en':
            synsets = wordnet.synsets(phrase)
            if synsets:
                lemmas = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
                if lemmas:
                    return random.choice(lemmas).replace('_', ' ')
        
        # For other languages, use translation-based synonyms
        return self.translation_cache[language].get(phrase)

    def _get_entity_spans(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Tuple[int, int, str]]:
        """Get entity spans with start, end, and type"""
        spans = []
        current_entity = None
        start_idx = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                if current_entity:
                    spans.append((start_idx, i, current_entity))
                current_entity = label[2:]
                start_idx = i
            elif label.startswith('I-'):
                continue
            else:
                if current_entity:
                    spans.append((start_idx, i, current_entity))
                    current_entity = None
        
        if current_entity:
            spans.append((start_idx, len(tokens), current_entity))
        
        return spans