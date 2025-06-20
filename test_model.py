"""
FloodSense: South Sudan - Model Testing
Tests the fine-tuned T5 model on sample queries.
"""
import os
import logging
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from infer import generate_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.Test")

# Constants
MODEL_DIR = "models/fine_tuned_t5"

def test_model():
    """Test the fine-tuned model on sample queries."""
    logger.info("Testing model on sample queries...")
    
    # Sample in-domain questions
    in_domain_questions = [
        "What is the flood risk in Bentiu?",
        "How can I prepare for floods?",
        "When is the flood season in South Sudan?",
        "What causes flooding in Juba?",
        "What safety measures should I take during a flood?",
        "How does climate change affect flooding in South Sudan?",
        "Where are evacuation centers located in Malakal?",
        "What emergency supplies should I have ready for floods?"
    ]
    
    # Sample out-of-domain questions
    out_of_domain_questions = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "Who won the World Cup in 2018?",
        "What are the best stocks to invest in?",
        "Can you recommend a good smartphone?",
        "What's the plot of Star Wars?",
        "How do I learn to play guitar?",
        "What's the weather like in New York today?"
    ]
    
    print("\n" + "="*50)
    print("TESTING IN-DOMAIN QUESTIONS")
    print("="*50)
    
    for question in in_domain_questions:
        print(f"\nQ: {question}")
        response = generate_response(question)
        print(f"A: {response}")
    
    print("\n" + "="*50)
    print("TESTING OUT-OF-DOMAIN QUESTIONS")
    print("="*50)
    
    for question in out_of_domain_questions:
        print(f"\nQ: {question}")
        response = generate_response(question)
        print(f"A: {response}")

def main():
    """Main function to run the model testing."""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_DIR):
            logger.error(f"Model directory not found: {MODEL_DIR}")
            logger.error("Please train the model first using train_model.py")
            return
        
        # Test model
        test_model()
        
        logger.info("Model testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model testing: {str(e)}")

if __name__ == "__main__":
    main()