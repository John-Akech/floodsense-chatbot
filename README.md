# FloodSense: South Sudan - Domain-Specific Chatbot

A specialized chatbot providing flood risk information for communities in South Sudan, built using transformer models.

## Project Overview

FloodSense is a domain-specific chatbot designed to provide critical flood risk information for communities in South Sudan. The chatbot uses a fine-tuned T5 transformer model to understand user queries and provide relevant, accurate responses about flood risks, preparation guidelines, emergency contacts, and more.

### Domain Focus: Flood Risk Management in South Sudan

This chatbot specializes in providing information about:
- Region-specific flood risk assessments
- Seasonal flooding patterns and predictions
- Emergency preparation guidelines
- Evacuation center locations
- Emergency contact information
- Climate impact insights

## Dataset

The chatbot is trained on a custom dataset of question-answer pairs related to flood risk in South Sudan. The dataset includes:
- 500+ question-answer pairs covering flood-related topics
- Region-specific flood risk information for 7 major regions
- Preparation and safety guidelines based on international standards
- Seasonal flood patterns derived from historical data
- Climate impact information from scientific reports

The dataset was created by combining:
1. Regional risk information
2. Expert knowledge about flood preparation and response
3. Generated question variations to improve model generalization

## Model Architecture

The chatbot uses a fine-tuned T5 (Text-to-Text Transfer Transformer) model from Hugging Face, implemented using TensorFlow. T5 was chosen for its:
- Strong performance on generative question-answering tasks
- Ability to handle both extractive and generative QA
- Efficient fine-tuning with limited data
- Reasonable inference speed for deployment

### Preprocessing

The data preprocessing pipeline includes:
- Tokenization using T5Tokenizer
- Input formatting as "question: {question}"
- Padding and truncation to appropriate sequence lengths
- Train-validation split (80%-20%)

### Domain-Specific Query Handling

The chatbot implements specialized domain detection to:
- Recognize and respond to flood-related queries
- Appropriately reject out-of-domain questions with Google search referrals
- Provide helpful guidance when users ask about topics outside its expertise

### Hyperparameter Tuning

Multiple hyperparameter configurations were tested to optimize performance:
| Configuration | Learning Rate | Batch Size | Epochs | Val Loss | Performance |
|---------------|--------------|------------|--------|----------|-------------|
| Config 1      | 1e-4         | 4          | 2      | 1.87     | Baseline    |
| Config 2      | 5e-5         | 8          | 3      | 1.62     | **Best**    |
| Config 3      | 3e-5         | 8          | 4      | 1.65     | Good        |

The best performance was achieved with Configuration 2 (learning rate: 5e-5, batch size: 8, epochs: 3), showing a 13.4% improvement over the baseline configuration.

## Evaluation

The model was evaluated using:
1. **ROUGE scores** - Measuring the quality of generated responses against reference answers
2. **BLEU score** - Evaluating the precision of generated text
3. **F1 score** - Assessing overall accuracy of responses
4. **Qualitative testing** - Manual review of responses for accuracy and helpfulness

### Results

**Training Performance:**
- Final Training Loss: 1.74
- Final Validation Loss: 0.72
- Training Time: 557 seconds (9.3 minutes)

**Evaluation Metrics:**
- **ROUGE-1**: 41.87
- **ROUGE-2**: 32.77
- **ROUGE-L**: 40.67
- **BLEU score**: 0.2747
- **F1 score**: 0.3747
- **Out-of-Domain Accuracy**: 100.0%

**Model Testing Results:**
✅ **In-Domain Queries**: Successfully answers flood-related questions with accurate, detailed responses  
✅ **Out-of-Domain Detection**: Perfect rejection of non-flood queries with appropriate referrals  
✅ **Hybrid System**: Combines rule-based responses with AI-generated answers for optimal performance  
✅ **Real-time Performance**: Fast inference with consistent response quality

The model successfully handles domain-specific queries while perfectly rejecting out-of-domain questions. The hybrid approach (rule-based + AI model) ensures reliable responses for flood-related queries and appropriate referrals for non-domain topics.

## User Interface

The chatbot is deployed as a Streamlit web application with a clean, intuitive interface:
1. **Chat Interface** - Interactive chatbot with message history
2. **History Tab** - Access to previous conversations
3. **File Upload** - Support for uploading documents
4. **Google Search Referral** - Automatic referral for out-of-domain questions

## Installation and Usage

1. Clone the repository
```bash
git clone https://github.com/username/floodsense-chatbot.git
cd floodsense-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

Note: This project requires Python 3.8+ and TensorFlow 2.12.0.

3. **Run the application (auto-setup on first run)**
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### 🚀 First Run Setup
**The chatbot will automatically:**
- Generate the flood risk dataset (500+ Q&A pairs)
- Train the T5 model (10-15 minutes)
- Start the interface when ready

### 📊 Manual Operations (Optional)
```bash
# Generate dataset only
python generate_dataset.py

# Train model only  
python train_model.py

# Evaluate performance
python evaluate_model.py

# Test model responses
python test_model.py
```

### ⚠️ Important Notes
- **First run takes 10-15 minutes** for model training
- **Subsequent runs start immediately**
- Use `streamlit run app.py` (not `python app.py`)
- Requires stable internet for model downloads

## Project Structure

```
floodrisk_chatbot/
├── data/                # Data files and processed datasets
│   ├── processed/       # Processed training and validation data
│   ├── flood_data.csv   # Regional flood risk information
│   └── qa_dataset.json  # Question-answer pairs dataset
├── models/              # Model storage directory
│   └── fine_tuned_t5/   # Fine-tuned T5 model files
├── chat_history/        # Saved chat history
├── logs/                # Application logs
├── app.py               # Main Streamlit application
├── evaluate_model.py    # Model evaluation script
├── fine_tune_model.py   # Hyperparameter tuning script
├── generate_dataset.py  # Dataset generation script
├── infer.py             # Model inference script
├── preprocess.py        # Data preprocessing script
├── test_model.py        # Model testing script
├── train_model.py       # Model training script
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Example Conversations

**User**: What is the flood risk in Bentiu?  
**FloodSense**: Bentiu has a High flood risk. The flood season typically runs from May to October, affecting approximately 120,000 people.

**User**: How can I prepare for floods?  
**FloodSense**: To prepare for floods:
1) Stay informed about weather forecasts and warnings
2) Prepare an emergency kit with food, water, and medicine
3) Know evacuation routes and safe shelter locations
4) Keep important documents in waterproof containers
5) Move to higher ground immediately when warnings are issued
6) Avoid walking or driving through flood waters
7) Disconnect electrical appliances if flooding is imminent

**User**: When is the flood season in South Sudan?  
**FloodSense**: The main flood season in South Sudan typically runs from May to October, with peak flooding usually occurring in August and September. The intensity and duration can vary by region.

**User**: What's the capital of France?  
**FloodSense**: I'm specialized in providing information about flood risks in South Sudan. I don't have information about that topic. You can try searching for this on Google: [Click here to search](https://www.google.com/search?q=What%27s+the+capital+of+France%3F)

## Features

- **Hybrid Response System**: Combines rule-based responses with AI-generated answers
- **Domain-Specific Knowledge**: Specialized in South Sudan flood information
- **Perfect Out-of-Domain Detection**: 100% accuracy in identifying non-flood queries
- **Google Search Referral**: Provides search links for non-flood related questions
- **File Upload**: Supports document uploads for context
- **Chat History**: Saves and loads previous conversations
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Inference**: Fast response generation using fine-tuned T5 model

## Testing & Validation

All core components have been tested and validated:

✅ **Model Training**: Successfully trained T5 model with decreasing loss (8.54 → 0.72)  
✅ **Model Evaluation**: Comprehensive metrics calculated using ROUGE, BLEU, and F1 scores  
✅ **Inference Pipeline**: Real-time response generation working correctly  
✅ **Domain Detection**: Perfect out-of-domain query handling (100% accuracy)  
✅ **User Interface**: Streamlit app fully functional with all features  
✅ **Data Pipeline**: Complete preprocessing and dataset handling