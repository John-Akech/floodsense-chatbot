"""
FloodSense: South Sudan - Data Preprocessing
Handles preprocessing of the Q&A dataset for model training.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.Preprocess")

def load_qa_dataset(dataset_path: str = "data/qa_dataset.json") -> List[Dict[str, str]]:
    """
    Load the Q&A dataset from JSON file.
    
    Args:
        dataset_path: Path to the dataset JSON file
        
    Returns:
        List of Q&A pairs
    """
    try:
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Dataset file not found: {dataset_path}")
            return []
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_for_t5(qa_pairs: List[Dict[str, str]], 
                      test_size: float = 0.2, 
                      random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Preprocess the Q&A pairs for T5 model training.
    
    Args:
        qa_pairs: List of Q&A pairs
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    try:
        logger.info(f"Preprocessing {len(qa_pairs)} Q&A pairs for T5 model")
        
        # Split into train and validation sets
        train_data, val_data = train_test_split(
            qa_pairs, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Split into {len(train_data)} training and {len(val_data)} validation examples")
        
        # Convert to datasets
        train_dataset = Dataset.from_dict({
            'question': [item['question'] for item in train_data],
            'answer': [item['answer'] for item in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'question': [item['question'] for item in val_data],
            'answer': [item['answer'] for item in val_data]
        })
        
        return train_dataset, val_dataset
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def analyze_dataset(qa_pairs: List[Dict[str, str]]) -> Dict:
    """
    Analyze the dataset to provide statistics.
    
    Args:
        qa_pairs: List of Q&A pairs
        
    Returns:
        Dictionary of statistics
    """
    try:
        # Extract questions and answers
        questions = [item['question'] for item in qa_pairs]
        answers = [item['answer'] for item in qa_pairs]
        
        # Calculate statistics
        question_lengths = [len(q.split()) for q in questions]
        answer_lengths = [len(a.split()) for a in answers]
        
        stats = {
            'total_examples': len(qa_pairs),
            'avg_question_length': np.mean(question_lengths),
            'max_question_length': np.max(question_lengths),
            'avg_answer_length': np.mean(answer_lengths),
            'max_answer_length': np.max(answer_lengths),
            'unique_questions': len(set(questions)),
            'unique_answers': len(set(answers))
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats
    
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise

def main():
    """Main function to run the preprocessing pipeline."""
    try:
        # Load dataset
        qa_pairs = load_qa_dataset()
        
        if not qa_pairs:
            logger.error("No data loaded. Exiting.")
            return
        
        # Analyze dataset
        stats = analyze_dataset(qa_pairs)
        
        # Preprocess for T5
        train_dataset, val_dataset = preprocess_for_t5(qa_pairs)
        
        logger.info(f"Created train dataset with {len(train_dataset)} examples")
        logger.info(f"Created validation dataset with {len(val_dataset)} examples")
        
        # Save processed datasets
        os.makedirs("data/processed", exist_ok=True)
        train_dataset.save_to_disk("data/processed/train")
        val_dataset.save_to_disk("data/processed/val")
        logger.info("Saved processed datasets to disk")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")

if __name__ == "__main__":
    main()