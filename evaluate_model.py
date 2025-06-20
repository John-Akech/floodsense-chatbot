"""
FloodSense: South Sudan - Model Evaluation
Evaluates the fine-tuned model on test examples and metrics.
"""
import os
import json
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from preprocess import load_qa_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.Evaluation")

# Constants
MODEL_DIR = "models/fine_tuned_t5"
DATASET_PATH = "data/qa_dataset.json"
RESULTS_PATH = "accuracy_results.csv"

def load_model_and_tokenizer(model_path: str = MODEL_DIR) -> tuple:
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        if os.path.exists(model_path):
            model = TFT5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            logger.warning(f"Model not found at {model_path}, using base T5-small")
            model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_examples(model: Any, tokenizer: Any, examples: List[Dict[str, str]], num_examples: int = None) -> List[Dict]:
    """
    Evaluate the model on specific examples.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        examples: List of examples to evaluate
        num_examples: Number of examples to evaluate (None for all)
        
    Returns:
        List of evaluation results
    """
    try:
        # Select examples
        if num_examples is not None:
            examples = examples[:num_examples]
        
        results = []
        
        for i, example in enumerate(examples):
            question = example["question"]
            reference_answer = example["answer"]
            
            # Prepare input
            input_text = f"question: {question}"
            input_encodings = tokenizer(
                input_text, 
                return_tensors="tf",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            # Generate answer
            outputs = model.generate(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated answer
            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate simple similarity score (word overlap)
            ref_words = set(reference_answer.lower().split())
            pred_words = set(predicted_answer.lower().split())
            
            if len(ref_words) > 0:
                precision = len(ref_words.intersection(pred_words)) / len(pred_words) if len(pred_words) > 0 else 0
                recall = len(ref_words.intersection(pred_words)) / len(ref_words)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0
            
            # Log the results
            result = {
                "question": question,
                "reference_answer": reference_answer,
                "predicted_answer": predicted_answer,
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "f1": round(f1 * 100, 2)
            }
            
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(examples)} examples")
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating examples: {str(e)}")
        raise

def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate aggregate metrics from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary of aggregate metrics
    """
    try:
        # Extract metrics
        precision_scores = [result["precision"] for result in results]
        recall_scores = [result["recall"] for result in results]
        f1_scores = [result["f1"] for result in results]
        
        # Calculate aggregate metrics
        metrics = {
            "avg_precision": np.mean(precision_scores),
            "avg_recall": np.mean(recall_scores),
            "avg_f1": np.mean(f1_scores),
            "median_precision": np.median(precision_scores),
            "median_recall": np.median(recall_scores),
            "median_f1": np.median(f1_scores),
            "min_precision": np.min(precision_scores),
            "min_recall": np.min(recall_scores),
            "min_f1": np.min(f1_scores),
            "max_precision": np.max(precision_scores),
            "max_recall": np.max(recall_scores),
            "max_f1": np.max(f1_scores)
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def save_results_to_csv(results: List[Dict], output_path: str = "evaluation_results.csv"):
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of evaluation results
        output_path: Path to save CSV
    """
    try:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results CSV to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving CSV results: {str(e)}")
        # Continue execution even if CSV saving fails

def evaluate_out_of_domain(model: Any, tokenizer: Any) -> Dict:
    """
    Evaluate the model's ability to handle out-of-domain queries.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Define out-of-domain questions
        out_of_domain_questions = [
            "What is the capital of France?",
            "How do I bake a chocolate cake?",
            "Who won the World Cup in 2018?",
            "What are the best stocks to invest in?",
            "Can you recommend a good smartphone?",
            "What's the plot of Star Wars?",
            "How do I learn to play guitar?",
            "What's the weather like in New York today?",
            "Tell me about quantum computing",
            "How do I fix my car's engine?"
        ]
        
        results = []
        
        for question in out_of_domain_questions:
            # Prepare input
            input_text = f"question: {question}"
            input_encodings = tokenizer(
                input_text, 
                return_tensors="tf",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            # Generate answer
            outputs = model.generate(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated answer
            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if the model correctly identified it as out-of-domain
            # Look for phrases indicating out-of-domain recognition
            out_of_domain_phrases = [
                "i don't know", "i'm not sure", "i don't have", "i can't answer",
                "i'm specialized", "i'm focused on", "flood risk", "south sudan"
            ]
            
            is_correct_response = any(phrase in predicted_answer.lower() for phrase in out_of_domain_phrases)
            
            results.append({
                "question": question,
                "response": predicted_answer,
                "correctly_identified": is_correct_response
            })
        
        # Calculate accuracy
        accuracy = sum(1 for r in results if r["correctly_identified"]) / len(results) * 100
        
        return {
            "out_of_domain_accuracy": accuracy,
            "examples": results
        }
    
    except Exception as e:
        logger.error(f"Error evaluating out-of-domain queries: {str(e)}")
        raise

def main():
    """Main function to run the evaluation pipeline."""
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Load dataset
        qa_pairs = load_qa_dataset(DATASET_PATH)
        
        if not qa_pairs:
            logger.error("No data loaded. Exiting.")
            return
        
        # Evaluate in-domain examples
        logger.info("Evaluating model on in-domain examples...")
        results = evaluate_examples(model, tokenizer, qa_pairs, num_examples=50)
        
        # Calculate metrics
        logger.info("Calculating aggregate metrics...")
        metrics = calculate_metrics(results)
        
        # Log metrics
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.2f}")
        
        # Evaluate out-of-domain handling
        logger.info("Evaluating out-of-domain query handling...")
        out_of_domain_results = evaluate_out_of_domain(model, tokenizer)
        logger.info(f"Out-of-domain accuracy: {out_of_domain_results['out_of_domain_accuracy']:.2f}%")
        
        # Add out-of-domain results to metrics
        metrics["out_of_domain_accuracy"] = out_of_domain_results["out_of_domain_accuracy"]
        
        # Save all results separately to avoid array length issues
        in_domain_path = f"{MODEL_DIR}/in_domain_results.json"
        out_domain_path = f"{MODEL_DIR}/out_domain_results.json"
        metrics_path = f"{MODEL_DIR}/metrics.json"
        
        # Save in-domain results
        with open(in_domain_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved in-domain results to {in_domain_path}")
        
        # Save out-of-domain results
        with open(out_domain_path, "w") as f:
            json.dump(out_of_domain_results["examples"], f, indent=2)
        logger.info(f"Saved out-of-domain results to {out_domain_path}")
        
        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Try to save CSV for in-domain results
        try:
            save_results_to_csv(results, RESULTS_PATH)
        except Exception as e:
            logger.warning(f"Could not save CSV: {str(e)}")
        
        logger.info("Evaluation completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")

if __name__ == "__main__":
    main()