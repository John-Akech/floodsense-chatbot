"""
FloodSense: South Sudan - Hyperparameter Tuning
Performs hyperparameter tuning for the T5 model using TensorFlow.
"""
import os
import json
import logging
import numpy as np
import tensorflow as tf
import time
from typing import Dict, List, Any
from datasets import Dataset
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from preprocess import load_qa_dataset, preprocess_for_t5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.FineTune")

# Constants
MODEL_NAME = "t5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
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

def train_and_evaluate(train_dataset: Dataset, val_dataset: Dataset, config: Dict) -> Dict:
    """
    Train and evaluate model with given hyperparameter configuration.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Hyperparameter configuration
        
    Returns:
        Dictionary of evaluation results
    """
    # Extract hyperparameters
    learning_rate = config["lr"]
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    
    logger.info(f"Training with config: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = TFT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Prepare TensorFlow datasets
    tf_train_dataset = prepare_tf_dataset(train_dataset, tokenizer, batch_size)
    tf_val_dataset = prepare_tf_dataset(val_dataset, tokenizer, batch_size)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create experiment directory
    config_dir = os.path.join(MODEL_DIR, f"config_{config['name'].replace(' ', '_')}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Custom training loop
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        epoch_loss = 0
        num_batches = 0
        
        for batch in tf_train_dataset:
            with tf.GradientTape() as tape:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    training=True
                )
                loss = outputs.loss
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
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
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                training=False
            )
            val_loss += outputs.loss.numpy()
            num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        logger.info(f"  Epoch {epoch+1}: Train Loss = {float(avg_train_loss):.4f}, Val Loss = {float(avg_val_loss):.4f}")
        
        # Save checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(os.path.join(config_dir, "best_model"))
            logger.info(f"  New best model saved with val_loss = {float(best_val_loss):.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save model for this configuration
    model.save_pretrained(config_dir)
    
    # Return evaluation results
    return {
        "config": config,
        "val_loss": float(best_val_loss),
        "training_time": training_time,
        "history": {
            "train_loss": [float(x) for x in train_losses],
            "val_loss": [float(x) for x in val_losses]
        }
    }

def run_hyperparameter_tuning():
    """
    Run hyperparameter tuning experiments.
    
    Returns:
        Best hyperparameter configuration
    """
    logger.info("Running hyperparameter tuning...")
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Using {len(gpus)} GPU(s)")
    else:
        logger.info("Using CPU")
    
    # Load dataset
    qa_pairs = load_qa_dataset(DATASET_PATH)
    
    # Preprocess data
    train_dataset, val_dataset = preprocess_for_t5(qa_pairs)
    
    # Define hyperparameter configurations
    configs = [
        {"name": "Config 1", "lr": 1e-4, "batch_size": 4, "epochs": 2},
        {"name": "Config 2", "lr": 5e-5, "batch_size": 8, "epochs": 3},
        {"name": "Config 3", "lr": 3e-5, "batch_size": 8, "epochs": 4}
    ]
    
    results = []
    
    # Run each configuration
    for config in configs:
        result = train_and_evaluate(train_dataset, val_dataset, config)
        results.append(result)
        logger.info(f"Results for {config['name']}: Loss = {result['val_loss']:.4f}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x["val_loss"])
    
    # Save results
    output_file = os.path.join(MODEL_DIR, "tuning_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "best_config": best_config
        }, f, indent=2)
    
    logger.info(f"Best configuration: {best_config['config']['name']} with loss = {best_config['val_loss']:.4f}")
    logger.info(f"Results saved to {output_file}")
    
    # Create a comparison table for the report
    comparison_table = {
        "config": [r["config"]["name"] for r in results],
        "learning_rate": [r["config"]["lr"] for r in results],
        "batch_size": [r["config"]["batch_size"] for r in results],
        "epochs": [r["config"]["epochs"] for r in results],
        "val_loss": [r["val_loss"] for r in results],
        "training_time": [r["training_time"] for r in results]
    }
    
    # Save comparison table
    with open(os.path.join(MODEL_DIR, "hyperparameter_comparison.json"), "w") as f:
        json.dump(comparison_table, f, indent=2)
    
    return best_config

def main():
    """Main function to run hyperparameter tuning."""
    try:
        # Create output directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Run hyperparameter tuning
        best_config = run_hyperparameter_tuning()
        
        logger.info("Hyperparameter tuning completed successfully!")
        logger.info(f"Best configuration: {best_config['config']}")
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")

if __name__ == "__main__":
    main()