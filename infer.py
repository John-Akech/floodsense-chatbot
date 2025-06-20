"""
FloodSense: South Sudan - Model Inference
Handles inference using the fine-tuned T5 model with TensorFlow.
"""
import os
import logging
import numpy as np
from typing import Optional, List

try:
    import tensorflow as tf
    from transformers import TFT5ForConditionalGeneration, T5Tokenizer
    TF_AVAILABLE = True
except (ImportError, RuntimeError):
    TF_AVAILABLE = False
    tf = None
    TFT5ForConditionalGeneration = None
    T5Tokenizer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.Inference")

# Define flood-related keywords for domain detection
FLOOD_KEYWORDS = [
    "flood", "floods", "flooding", "water level", "evacuation", "rain", "rainfall",
    "south sudan", "bentiu", "bor", "malakal", "juba", "tonj", "yei", "wau",
    "emergency", "preparation", "safety", "risk", "warning", "season", "climate",
    "change", "warming", "shelter", "center", "centres", "global", "weather",
    "regional assessments", "safety guidelines", "climate information",
    "hello", "hi", "hey", "greetings", "jonglei", "johnglei", "upper nile",
    "unity state", "equatoria", "region", "county", "state", "payam", "boma",
    "central equatoria", "eastern equatoria", "western equatoria", "northern bahr el ghazal",
    "western bahr el ghazal", "lakes", "warrap", "aweil", "rumbek", "kuajok",
    "torit", "kapoeta", "magwi", "pochalla", "pibor", "akobo", "nasir",
    "melut", "renk", "kodok", "fashoda", "maban", "pariang", "rubkona",
    "mayom", "koch", "leer", "panyijiar", "guit", "mayendit", "abiemnhom"
]

class FloodRiskModel:
    """Class for handling inference with the fine-tuned T5 model using TensorFlow."""
    
    def __init__(self, model_path: Optional[str] = "models/fine_tuned_t5"):
        """
        Initialize the model for inference.
        
        Args:
            model_path: Path to the fine-tuned model. If None, uses the base T5-small model.
        """
        if not TF_AVAILABLE:
            self.model = None
            self.tokenizer = None
            return
            
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Using {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU memory config error: {str(e)}")
        else:
            logger.info("Using CPU")
        
        try:
            # Load model and tokenizer
            model_files_exist = model_path and os.path.exists(model_path) and any(
                f.endswith('.h5') for f in os.listdir(model_path)
            ) if model_path and os.path.exists(model_path) else False
            
            if model_files_exist:
                logger.info(f"Loading fine-tuned model from {model_path}")
                self.model = TFT5ForConditionalGeneration.from_pretrained(model_path)
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            else:
                logger.warning("Fine-tuned model not found or incomplete. Using base T5-small model.")
                self.model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
                self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to rule-based only
            logger.info("Falling back to rule-based responses only")
            self.model = None
            self.tokenizer = None
    
    def is_in_domain(self, query: str, threshold: float = 0.1) -> bool:
        """
        Check if the query is within the flood risk domain.
        
        Args:
            query: The user's question
            threshold: Minimum keyword match ratio to consider in-domain
            
        Returns:
            Boolean indicating if query is in-domain
        """
        query_lower = query.lower()
        
        # Check for flood/climate keywords with any South Sudan location
        flood_climate_words = ["climate change", "flood", "floods", "flooding", "rain", "rainfall", "water level"]
        location_words = ["south sudan", "jonglei", "johnglei", "upper nile", "unity state", "equatoria", 
                         "region", "county", "state", "payam", "boma", "central equatoria", "eastern equatoria", 
                         "western equatoria", "northern bahr el ghazal", "western bahr el ghazal", "lakes", 
                         "warrap", "bentiu", "bor", "malakal", "juba", "tonj", "yei", "wau", "aweil", 
                         "rumbek", "kuajok", "torit", "kapoeta", "magwi", "pochalla", "pibor", "akobo", 
                         "nasir", "melut", "renk", "kodok", "fashoda", "maban", "pariang", "rubkona", 
                         "mayom", "koch", "leer", "panyijiar", "guit", "mayendit", "abiemnhom"]
        
        has_flood_climate = any(word in query_lower for word in flood_climate_words)
        has_location = any(word in query_lower for word in location_words)
        
        if has_flood_climate and has_location:
            return True
            
        # Check for any other flood-related keywords
        return any(keyword.lower() in query_lower for keyword in FLOOD_KEYWORDS)
    
    def generate_response(self, query: str, max_length: int = 150) -> str:
        """
        Generate a response for the given query.
        
        Args:
            query: The user's question
            max_length: Maximum length of the generated response
            
        Returns:
            Generated response text
        """
        try:
            # Check if query is in domain
            if not self.is_in_domain(query):
                return "I'm sorry, I'm specialized in flood risk information for South Sudan. I don't have information about that topic. Could you ask me something about flood risks, preparation, or safety in South Sudan?"
            
            # Use rule-based responses for common questions
            rule_based_response = self.get_rule_based_response(query)
            if rule_based_response:
                return rule_based_response
            
            # If TensorFlow model is not available, use fallback response
            if not TF_AVAILABLE or self.model is None:
                return "I can help with flood information in South Sudan. Please ask about specific regions (Bentiu, Bor, Malakal, Juba), flood preparation, safety measures, or seasonal patterns."
            
            # Format input as expected by T5
            input_text = f"question: {query}"
            
            # Tokenize input
            input_encodings = self.tokenizer(
                input_text, 
                return_tensors="tf",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            # Generate response
            output = self.model.generate(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            # Decode response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # If response is too short or doesn't make sense, use a fallback
            if len(response.split()) < 5:
                return "I'm still learning about flood risks in South Sudan. Could you please rephrase your question or ask about flood preparation, safety, or specific regions?"
            
            logger.info(f"Generated response for query: '{query}'")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."
            
    def get_rule_based_response(self, query: str) -> str:
        """
        Get rule-based response for common questions.
        
        Args:
            query: The user's question
            
        Returns:
            Rule-based response or None if no rule matches
        """
        query = query.lower().strip()
        
        # Check for greetings
        if query in ["hello", "hi", "hey", "greetings"] or any(greeting in query for greeting in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm FloodSense, your assistant for flood information in South Sudan. How can I help you today?"
        
        # Check for comprehensive information requests
        if "regional assessments" in query:
            return """FLOOD RISK ASSESSMENTS BY REGION:

HIGH RISK REGIONS:
• Bentiu: High flood risk, May-October season, affects ~120,000 people
• Bor: High flood risk, May-October season, affects ~95,000 people  
• Malakal: High flood risk, May-October season, affects ~110,000 people

MEDIUM RISK REGIONS:
• Juba: Medium flood risk, June-September season, affects ~75,000 people
• Tonj: Medium flood risk, June-September season, affects ~45,000 people

LOW RISK REGIONS:
• Yei: Low flood risk, July-September season, affects ~30,000 people
• Wau: Low flood risk, July-August season, affects ~25,000 people

All regions experience seasonal flooding with varying intensity and duration."""
        
        if "safety guidelines" in query:
            return """COMPREHENSIVE FLOOD SAFETY GUIDELINES:

PREPARATION:
• Stay informed about weather forecasts and warnings
• Prepare emergency kit with food, water, medicine (3-day supply)
• Know evacuation routes and safe shelter locations
• Keep important documents in waterproof containers
• Identify higher ground locations in your area

DURING FLOODS:
• Move to higher ground immediately when warnings issued
• Never walk, swim, or drive through flood waters
• Stay off bridges over fast-moving water
• Evacuate if told to do so by authorities
• Stay away from downed power lines
• Disconnect electrical appliances if flooding imminent

AFTER FLOODS:
• Return home only when authorities say it's safe
• Check for structural damage before entering buildings
• Avoid contaminated flood water
• Boil water before drinking if water supply affected
• Report damaged utilities to authorities"""
        
        if "climate information" in query:
            return """CLIMATE AND SEASONAL INFORMATION:

FLOOD SEASONS:
• Main season: May to October (peak: August-September)
• Varies by region: Northern areas start earlier, Southern areas later
• Duration and intensity vary annually

RAINFALL PATTERNS:
• Heavy seasonal rainfall during wet season
• Unpredictable weather patterns due to climate change
• CHIRPS satellite data helps monitor precipitation

CLIMATE CHANGE IMPACTS:
• Increased rainfall intensity and unpredictable patterns
• More frequent extreme weather events
• Changes in seasonal rainfall distribution
• Rising temperatures increase evaporation and precipitation
• Altered White Nile river flow patterns
• Prolonged droughts followed by intense flooding
• Environmental degradation reduces natural flood defenses
• Makes flood prediction more difficult

RIVER SYSTEMS:
• White Nile overflow is major flood cause
• Tributary systems contribute to regional flooding
• Poor drainage infrastructure worsens urban flooding"""
        
        # Check for region-specific questions
        if any(word in query for word in ["jonglei", "johnglei"]):
            return "Jonglei State has a Very High flood risk. The flood season runs from May to November, with severe flooding affecting over 800,000 people annually. The White Nile and Sobat River systems cause extensive seasonal flooding across the state."
        
        if "upper nile" in query:
            return "Upper Nile State has a High flood risk. Seasonal flooding occurs from June to October, affecting approximately 600,000 people. The White Nile and tributaries cause widespread flooding in Malakal, Melut, and surrounding areas."
        
        if "unity state" in query:
            return "Unity State has a Very High flood risk. Flooding occurs from May to November, affecting over 700,000 people. Bentiu and surrounding areas experience severe seasonal flooding from White Nile overflow."
        
        if any(word in query for word in ["equatoria", "central equatoria", "eastern equatoria", "western equatoria"]):
            return "Equatoria regions have Medium to Low flood risk. Central Equatoria (including Juba) has medium risk from June-September. Eastern and Western Equatoria have lower risks with localized flooding during heavy rains."
        
        regions = ["bentiu", "bor", "malakal", "juba", "tonj", "yei", "wau"]
        for region in regions:
            if region in query:
                if region == "bentiu":
                    return "Bentiu has a High flood risk. The flood season typically runs from May to October, affecting approximately 120,000 people."
                elif region == "bor":
                    return "Bor has a High flood risk. The flood season typically runs from May to October, affecting approximately 95,000 people."
                elif region == "malakal":
                    return "Malakal has a High flood risk. The flood season typically runs from May to October, affecting approximately 110,000 people."
                elif region == "juba":
                    return "Juba has a Medium flood risk. The flood season typically runs from June to September, affecting approximately 75,000 people."
                elif region == "tonj":
                    return "Tonj has a Medium flood risk. The flood season typically runs from June to September, affecting approximately 45,000 people."
                elif region == "yei":
                    return "Yei has a Low flood risk. The flood season typically runs from July to September, affecting approximately 30,000 people."
                elif region == "wau":
                    return "Wau has a Low flood risk. The flood season typically runs from July to August, affecting approximately 25,000 people."
        
        # Check for evacuation center questions
        if any(word in query for word in ["evacuation", "center", "centres", "shelter"]) and any(region in query for region in ["malakal", "bentiu", "bor", "juba"]):
            return """Evacuation centers are typically located at:
1) Schools and community centers on higher ground
2) Government buildings and administrative offices
3) Religious facilities (churches, mosques)
4) UN and NGO compounds when available
5) Designated safe areas identified by local authorities
Contact local authorities or humanitarian organizations for specific locations during flood warnings."""
        
        # Check for climate change questions
        if any(word in query for word in ["climate", "change", "global", "warming"]) and any(word in query for word in ["flood", "floods", "flooding"]):
            return """Climate change affects flooding in South Sudan through:
1) Increased rainfall intensity and unpredictable weather patterns
2) More frequent extreme weather events
3) Changes in seasonal rainfall distribution
4) Rising temperatures leading to increased evaporation and precipitation
5) Altered river flow patterns affecting the White Nile system
6) Prolonged droughts followed by intense flooding
7) Environmental degradation reducing natural flood defenses
These changes make flood prediction more difficult and increase vulnerability of communities."""
        
        # Check for preparation questions
        if any(word in query for word in ["prepare", "preparation", "preparing", "ready"]):
            return """To prepare for floods:
1) Stay informed about weather forecasts and warnings
2) Prepare an emergency kit with food, water, and medicine
3) Know evacuation routes and safe shelter locations
4) Keep important documents in waterproof containers
5) Move to higher ground immediately when warnings are issued
6) Avoid walking or driving through flood waters
7) Disconnect electrical appliances if flooding is imminent"""
        
        # Check for safety questions
        if any(word in query for word in ["safety", "safe", "protect", "protection"]) and any(word in query for word in ["flood", "floods", "flooding"]):
            return """Flood safety tips:
1) Never walk, swim, or drive through flood waters
2) Stay off bridges over fast-moving water
3) Evacuate if told to do so
4) Move to higher ground or a higher floor
5) Stay away from downed power lines
6) Return home only when authorities say it's safe"""
        
        # Check for causes questions
        if any(word in query for word in ["cause", "causes", "why", "reason"]) and any(word in query for word in ["flood", "floods", "flooding"]):
            return """Floods in South Sudan are primarily caused by:
1) Heavy seasonal rainfall during the wet season (May-October)
2) Overflow of the White Nile and its tributaries
3) Poor drainage infrastructure in urban areas
4) Deforestation and land degradation reducing water absorption
5) Climate change leading to more intense rainfall patterns"""
        
        # Check for season questions
        if any(word in query for word in ["season", "when", "time", "period"]) and any(word in query for word in ["flood", "floods", "flooding"]):
            return "The main flood season in South Sudan typically runs from May to October, with peak flooding usually occurring in August and September. The intensity and duration can vary by region."
        
        # No rule matched
        return None

# Singleton instance for reuse
_model_instance = None

def get_model(model_path: Optional[str] = "models/fine_tuned_t5") -> FloodRiskModel:
    """
    Get or create the model instance.
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        FloodRiskModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        # Auto-setup if model doesn't exist
        if not os.path.exists(model_path):
            print("Setting up FloodSense for first run...")
            from generate_dataset import create_qa_dataset
            from train_model import main as train_main
            
            # Generate dataset
            print("Generating dataset...")
            create_qa_dataset()
            
            # Train model
            print("Training model (this may take 10-15 minutes)...")
            train_main()
            
        _model_instance = FloodRiskModel(model_path)
    
    return _model_instance

def generate_response(query: str) -> str:
    """
    Generate a response for the given query using the model.
    
    Args:
        query: The user's question
        
    Returns:
        Generated response text
    """
    try:
        # Get model instance
        model = get_model()
        
        # Generate response
        return model.generate_response(query)
    
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return "I'm sorry, I couldn't process your question at this time."

def main():
    """Main function to demonstrate model inference."""
    # Test in-domain questions
    in_domain_questions = [
        "What is the flood risk in Bentiu?",
        "How can I prepare for floods?",
        "When is the flood season in South Sudan?",
        "What causes flooding in Juba?"
    ]
    
    # Test out-of-domain questions
    out_of_domain_questions = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "Who won the World Cup in 2018?",
        "What are the best stocks to invest in?"
    ]
    
    print("Testing model inference...")
    
    print("\n--- In-Domain Questions ---")
    for question in in_domain_questions:
        print(f"\nQ: {question}")
        response = generate_response(question)
        print(f"A: {response}")
    
    print("\n--- Out-of-Domain Questions ---")
    for question in out_of_domain_questions:
        print(f"\nQ: {question}")
        response = generate_response(question)
        print(f"A: {response}")

if __name__ == "__main__":
    main()