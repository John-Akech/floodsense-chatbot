"""
FloodSense: South Sudan - Model Training
Fine-tunes a T5 model for the flood risk Q&A task using TensorFlow.
"""
import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from datasets import Dataset
import time
import sys
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.Training")

# Check for required dependencies
try:
    import sentencepiece
except ImportError:
    logger.info("SentencePiece not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
    logger.info("SentencePiece installed successfully.")

from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from generate_dataset import create_qa_dataset
from preprocess import preprocess_for_t5, load_qa_dataset

# Constants
MODEL_NAME = "t5-small"  # Using T5-small for efficiency
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64
MODEL_DIR = "models/fine_tuned_t5"
DATASET_PATH = "data/qa_dataset.json"

def prepare_tf_dataset(dataset: Dataset, tokenizer: T5Tokenizer, batch_size: int = 32) -> tf.data.Dataset:
    """
    Prepare TensorFlow dataset from Hugging Face dataset.
    
    Args:
        dataset: Hugging Face dataset
        tokenizer: T5 tokenizer
        batch_size: Batch size for training
        
    Returns:
        TensorFlow dataset
    """
    # Extract questions and answers
    questions = dataset["question"]
    answers = dataset["answer"]
    
    # Format inputs for T5
    inputs = ["question: " + q for q in questions]
    
    # Tokenize inputs
    input_encodings = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    
    # Tokenize targets
    target_encodings = tokenizer(
        answers,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    
    # Convert to TensorFlow tensors
    input_ids = tf.convert_to_tensor(input_encodings["input_ids"])
    attention_mask = tf.convert_to_tensor(input_encodings["attention_mask"])
    labels = tf.convert_to_tensor(target_encodings["input_ids"])
    
    # Create dataset
    dataset_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    # Create TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
    
    # Shuffle and batch
    tf_dataset = tf_dataset.shuffle(len(questions)).batch(batch_size)
    
    return tf_dataset

def train_model(train_dataset: Dataset, val_dataset: Dataset, 
                learning_rate: float = 5e-5, batch_size: int = 2, 
                num_epochs: int = 5) -> Tuple[Any, Any]:
    """
    Fine-tune the T5 model on the flood risk Q&A dataset using TensorFlow.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Training model with {MODEL_NAME}...")
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = TFT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Prepare TensorFlow datasets
    tf_train_dataset = prepare_tf_dataset(train_dataset, tokenizer, batch_size)
    tf_val_dataset = prepare_tf_dataset(val_dataset, tokenizer, batch_size)
    
    # Create optimizer with better settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Custom training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        epoch_loss = 0
        num_batches = 0
        
        for batch in tf_train_dataset:
            try:
                with tf.GradientTape() as tape:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        training=True
                    )
                    loss = outputs.loss
                    # Ensure loss is a scalar
                    if len(loss.shape) > 0:
                        loss = tf.reduce_mean(loss)
                    
                gradients = tape.gradient(loss, model.trainable_variables)
                # Filter out None gradients
                gradients = [g if g is not None else tf.zeros_like(v) 
                           for g, v in zip(gradients, model.trainable_variables)]
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            except Exception as e:
                logger.warning(f"Batch training error: {e}. Skipping batch.")
                continue
            
            epoch_loss += loss.numpy()
            num_batches += 1
            
            if num_batches % 5 == 0:
                logger.info(f"  Batch {num_batches}: Loss = {float(loss.numpy()):.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = 0
        num_val_batches = 0
        
        for batch in tf_val_dataset:
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    training=False
                )
                val_loss += outputs.loss.numpy()
                num_val_batches += 1
            except Exception as e:
                logger.warning(f"Validation batch error: {e}. Skipping batch.")
                continue
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        logger.info(f"  Epoch {epoch+1}: Train Loss = {float(avg_train_loss):.4f}, Val Loss = {float(avg_val_loss):.4f}")
        
        # Save checkpoint
        model.save_weights(os.path.join(MODEL_DIR, f"checkpoint-{epoch+1}"))
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    logger.info(f"Model saved to {MODEL_DIR}")
    
    # Save training history
    history_dict = {
        "train_loss": [float(x) for x in train_losses],
        "val_loss": [float(x) for x in val_losses]
    }
    
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)
    
    return model, tokenizer

def main():
    """Main function to run the model training pipeline."""
    try:
        # Set memory growth for GPUs and configure TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Using {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        else:
            logger.info("Using CPU")
        
        # Configure TensorFlow for stability
        tf.config.optimizer.set_jit(False)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Load or create dataset
        if os.path.exists(DATASET_PATH):
            qa_pairs = load_qa_dataset(DATASET_PATH)
        else:
            qa_pairs = create_qa_dataset(DATASET_PATH)
        
        # Preprocess data
        train_dataset, val_dataset = preprocess_for_t5(qa_pairs)
        
        # Train model
        model, tokenizer = train_model(
            train_dataset, 
            val_dataset
        )
        
        logger.info("Model training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")

if __name__ == "__main__":
    main()