"""
FloodSense: South Sudan - Dataset Generation
Creates a comprehensive Q&A dataset for fine-tuning the transformer model.
"""
import os
import json
import logging
import pandas as pd
import random
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FloodSense.DataGen")

def load_flood_data(data_path: str = "data/flood_data.csv") -> pd.DataFrame:
    """
    Load the flood risk data from CSV.
    
    Args:
        data_path: Path to flood data CSV
        
    Returns:
        DataFrame with flood data
    """
    try:
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            logger.warning(f"Data file not found: {data_path}")
            # Create sample data
            data = {
                "region": ["Bentiu", "Bor", "Malakal", "Juba", "Tonj", "Yei", "Wau"],
                "risk_level": ["High", "High", "High", "Medium", "Medium", "Low", "Low"],
                "flood_season_start": ["May", "May", "May", "June", "June", "July", "July"],
                "flood_season_end": ["October", "October", "October", "September", "September", "September", "August"],
                "population_affected": [120000, 95000, 110000, 75000, 45000, 30000, 25000]
            }
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def generate_question_variations(base_question: str, region: str = None) -> List[str]:
    """
    Generate variations of a question.
    
    Args:
        base_question: Base question template
        region: Region name to insert (if applicable)
        
    Returns:
        List of question variations
    """
    variations = []
    
    # Replace region placeholder if provided
    if region:
        question = base_question.replace("{region}", region)
    else:
        question = base_question
    
    # Add the base question
    variations.append(question)
    
    # Generate variations with different phrasing
    if "what is" in question.lower():
        variations.append(question.lower().replace("what is", "tell me about"))
        variations.append(question.lower().replace("what is", "could you explain"))
    
    if "how can" in question.lower():
        variations.append(question.lower().replace("how can", "what's the best way to"))
        variations.append(question.lower().replace("how can", "what should i do to"))
    
    if "when" in question.lower():
        variations.append(question.lower().replace("when", "what time of year"))
        variations.append(question.lower().replace("when", "during which months"))
    
    if "why" in question.lower():
        variations.append(question.lower().replace("why", "what causes"))
        variations.append(question.lower().replace("why", "what are the reasons"))
    
    # Add question marks if missing
    variations = [v if v.endswith("?") else v + "?" for v in variations]
    
    # Capitalize first letter
    variations = [v[0].upper() + v[1:] if v else v for v in variations]
    
    return variations

def create_qa_dataset(output_path: str = "data/qa_dataset.json") -> List[Dict[str, str]]:
    """
    Create a comprehensive Q&A dataset for flood risk information.
    
    Args:
        output_path: Path to save the generated dataset
        
    Returns:
        List of Q&A pairs
    """
    try:
        # Load flood data
        flood_data = load_flood_data()
        
        qa_pairs = []
        regions = flood_data["region"].tolist()
        
        # 1. Region-specific risk questions
        for _, row in flood_data.iterrows():
            region = row["region"]
            risk_level = row["risk_level"]
            season_start = row["flood_season_start"]
            season_end = row["flood_season_end"]
            population = row["population_affected"]
            
            # Risk level questions
            base_question = f"What is the flood risk in {region}?"
            answer = f"{region} has a {risk_level} flood risk. The flood season typically runs from {season_start} to {season_end}, affecting approximately {population:,} people."
            
            variations = generate_question_variations(base_question)
            variations.extend([
                f"How severe are floods in {region}?",
                f"Is {region} prone to flooding?",
                f"What's the flooding situation like in {region}?"
            ])
            
            for question in variations:
                qa_pairs.append({"question": question, "answer": answer})
            
            # Season questions
            season_question = f"When is the flood season in {region}?"
            season_answer = f"The flood season in {region} typically runs from {season_start} to {season_end}."
            
            season_variations = generate_question_variations(season_question)
            season_variations.extend([
                f"What months does {region} experience flooding?",
                f"When should I be prepared for floods in {region}?",
                f"What is the flooding period in {region}?"
            ])
            
            for question in season_variations:
                qa_pairs.append({"question": question, "answer": season_answer})
            
            # Population questions
            pop_question = f"How many people are affected by floods in {region}?"
            pop_answer = f"Approximately {population:,} people are affected by flooding in {region}."
            
            pop_variations = generate_question_variations(pop_question)
            pop_variations.extend([
                f"What is the population impact of floods in {region}?",
                f"How many residents are at risk from flooding in {region}?"
            ])
            
            for question in pop_variations:
                qa_pairs.append({"question": question, "answer": pop_answer})
        
        # 2. Country Assessment (60 questions)
        country_questions = [
            ("What are the main causes of flooding in South Sudan?", "Flooding in South Sudan is caused by: 1) Heavy seasonal rainfall (May-October), 2) White Nile and tributary overflow, 3) Poor drainage infrastructure, 4) Deforestation reducing water absorption, 5) Climate change intensifying rainfall patterns, 6) Flat terrain in flood-prone areas."),
            ("How does the government coordinate flood response?", "The government coordinates through: 1) National Disaster Management Authority, 2) State-level emergency committees, 3) Early warning systems, 4) UN/NGO coordination, 5) Emergency funding allocation, 6) Evacuation planning with local authorities."),
            ("What is South Sudan's national flood risk mapping?", "National flood risk mapping uses: 1) SRTM elevation data, 2) Sentinel-1 satellite imagery, 3) CHIRPS rainfall data, 4) Historical flood records, 5) Population density analysis, 6) Infrastructure vulnerability assessments."),
            ("How does climate change impact flooding nationally?", "Climate change affects flooding through: 1) Increased rainfall intensity, 2) Unpredictable weather patterns, 3) Extended wet seasons, 4) More frequent extreme events, 5) Rising temperatures, 6) Altered river flow patterns."),
            ("What disaster financing exists for floods?", "Disaster financing includes: 1) Government emergency reserves, 2) World Bank disaster risk financing, 3) UN emergency response funds, 4) NGO humanitarian funding, 5) Community savings, 6) International donor support."),
            ("What are the long-term national effects of flooding?", "Long-term effects include: 1) Mass displacement of populations, 2) Deepened poverty cycles, 3) Infrastructure degradation, 4) Economic losses, 5) Food insecurity, 6) Reduced development progress."),
            ("How frequent are floods in South Sudan?", "Floods occur annually during wet season (May-October), with major floods every 2-3 years. Severity varies by region, with Jonglei, Unity, and Upper Nile experiencing the most frequent flooding."),
            ("What is the national flood early warning system?", "The system includes: 1) Meteorological monitoring, 2) River level gauges, 3) Satellite observations, 4) Community-based reporting, 5) Radio broadcasts, 6) Mobile alerts in accessible areas."),
            ("How do floods affect national food security?", "Floods impact food security by: 1) Destroying crops, 2) Disrupting markets, 3) Limiting access to food, 4) Increasing prices, 5) Reducing livestock, 6) Creating dependency on food aid."),
            ("What is the government's flood preparedness strategy?", "The strategy involves: 1) Pre-positioning relief supplies, 2) Training emergency responders, 3) Community awareness campaigns, 4) Infrastructure improvements, 5) Coordination mechanisms, 6) Contingency planning.")
        ]
        
        for question, answer in country_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:2]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 3. Regional Assessments (60 questions)
        regional_questions = [
            ("What is the flood risk in Jonglei State?", "Jonglei has EXTREME flood risk due to: 1) Flat terrain, 2) White Nile proximity, 3) Poor drainage, 4) High rainfall, 5) Limited infrastructure. Affects 800,000+ people annually."),
            ("What is the flood risk in Unity State?", "Unity has VERY HIGH flood risk due to: 1) Oil field flooding, 2) River overflow, 3) Poor road access, 4) Wetland areas. Affects 600,000+ people with limited evacuation routes."),
            ("What is the flood risk in Upper Nile State?", "Upper Nile has HIGH flood risk from: 1) Blue and White Nile confluence, 2) Seasonal flooding, 3) Refugee populations, 4) Limited infrastructure. Affects 500,000+ people annually."),
            ("How do rural and urban flood risks compare?", "Rural areas face: 1) Higher exposure, 2) Limited early warning, 3) Poor evacuation routes, 4) Agricultural losses. Urban areas have: 1) Better services, 2) Drainage issues, 3) Dense populations, 4) Infrastructure damage."),
            ("What are regional early warning gaps?", "Gaps include: 1) Limited weather stations, 2) Poor communication networks, 3) Language barriers, 4) Remote area coverage, 5) Technical capacity, 6) Community awareness."),
            ("Which regions have the highest flood mortality?", "Highest mortality in: 1) Jonglei State, 2) Unity State, 3) Upper Nile, 4) Lakes State, due to extreme flooding, poor access to healthcare, and limited evacuation capacity."),
            ("What makes Bentiu particularly vulnerable?", "Bentiu vulnerability: 1) Oil infrastructure, 2) Displaced populations, 3) Poor drainage, 4) Limited high ground, 5) Seasonal access roads, 6) Overcrowding in camps."),
            ("How does flooding affect Bor town?", "Bor experiences: 1) Annual White Nile flooding, 2) Airport closure, 3) Market disruption, 4) Health facility damage, 5) Population displacement, 6) Economic losses."),
            ("What is the flood risk in Lakes State?", "Lakes State has MODERATE-HIGH risk from: 1) Seasonal rainfall, 2) Poor drainage, 3) Cattle camp flooding, 4) Limited infrastructure. Affects pastoralist communities significantly."),
            ("How do cross-border floods affect regions?", "Cross-border flooding from Uganda and Ethiopia affects: 1) Upper Nile regions, 2) Refugee settlements, 3) Cross-border trade, 4) Regional coordination challenges.")
        ]
        
        for question, answer in regional_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:2]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 4. Climate & Hydrology (50 questions)
        climate_questions = [
            ("What are South Sudan's rainfall patterns?", "Rainfall patterns: 1) Wet season May-October, 2) Peak July-September, 3) 600-1200mm annually, 4) North-south gradient, 5) High variability, 6) Climate change impacts increasing intensity."),
            ("How do CHIRPS satellite observations help?", "CHIRPS data provides: 1) Rainfall estimates, 2) Drought monitoring, 3) Flood forecasting, 4) Climate trend analysis, 5) Early warning inputs, 6) Agricultural planning support."),
            ("What causes White Nile river overflow?", "White Nile overflow caused by: 1) Upstream rainfall in Uganda, 2) Lake Victoria levels, 3) Sudd wetland capacity, 4) Channel blockages, 5) Seasonal flow patterns, 6) Climate variability."),
            ("How does climate change intensify floods?", "Climate change intensifies through: 1) Extreme rainfall events, 2) Temperature increases, 3) Altered seasonal patterns, 4) More frequent droughts and floods, 5) Ecosystem changes."),
            ("What is the Sudd wetland's role in flooding?", "The Sudd: 1) Regulates river flow, 2) Stores floodwater, 3) Affects downstream flooding, 4) Supports biodiversity, 5) Influences local climate, 6) Provides flood protection when healthy.")
        ]
        
        for question, answer in climate_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:3]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 5. Safety & Preparedness (60 questions)
        safety_questions = [
            ("How should households prepare for floods?", "Household preparation: 1) Create emergency kit, 2) Identify evacuation routes, 3) Store important documents, 4) Plan family meeting points, 5) Monitor weather warnings, 6) Prepare elevated storage."),
            ("What are emergency evacuation procedures?", "Evacuation procedures: 1) Listen to warnings, 2) Move to higher ground, 3) Follow designated routes, 4) Bring emergency supplies, 5) Help vulnerable neighbors, 6) Report to evacuation centers."),
            ("What should be in a flood survival kit?", "Survival kit contents: 1) Water (3 days supply), 2) Non-perishable food, 3) First aid supplies, 4) Flashlight and batteries, 5) Radio, 6) Medications, 7) Important documents, 8) Cash."),
            ("How to ensure safe water during floods?", "Safe water practices: 1) Boil water for 1 minute, 2) Use water purification tablets, 3) Store in clean containers, 4) Avoid floodwater contact, 5) Use bottled water when available."),
            ("How to protect vulnerable populations?", "Protect vulnerable groups by: 1) Priority evacuation, 2) Special medical needs, 3) Accessible shelters, 4) Nutritional support, 5) Psychological care, 6) Family reunification."),
            ("What are flood safety rules for children?", "Child safety rules: 1) Never play in floodwater, 2) Stay with adults, 3) Learn evacuation routes, 4) Know emergency contacts, 5) Understand warning signals, 6) Practice safety drills."),
            ("How to prepare pregnant women for floods?", "Pregnancy preparation: 1) Medical records ready, 2) Emergency delivery kit, 3) Priority evacuation, 4) Prenatal medication supply, 5) Healthcare provider contact, 6) Birth plan backup."),
            ("What sanitation practices prevent disease?", "Sanitation practices: 1) Proper waste disposal, 2) Hand washing, 3) Food safety, 4) Clean water use, 5) Latrine maintenance, 6) Personal hygiene, 7) Vector control."),
            ("How to create community early warning?", "Community warning systems: 1) Local observers, 2) Communication networks, 3) Warning signals, 4) Evacuation plans, 5) Community drills, 6) Traditional knowledge integration."),
            ("What are flood-safe building practices?", "Safe building: 1) Elevated foundations, 2) Flood-resistant materials, 3) Proper drainage, 4) Emergency exits, 5) Utility protection, 6) Local building codes compliance.")
        ]
        
        for question, answer in safety_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:2]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 6. Infrastructure Impacts (50 questions)
        infrastructure_questions = [
            ("How often do roads collapse during floods?", "Road damage occurs: 1) 70% of rural roads affected annually, 2) Bridge failures common, 3) Impassable for weeks, 4) Emergency repairs needed, 5) Economic losses significant."),
            ("How do floods affect power and communication?", "Infrastructure impacts: 1) Power outages widespread, 2) Cell towers damaged, 3) Internet disruption, 4) Radio stations affected, 5) Emergency communication challenges."),
            ("What happens to schools and clinics?", "Facility impacts: 1) 40% of schools close during floods, 2) Health clinics inaccessible, 3) Equipment damage, 4) Staff displacement, 5) Service interruption for months."),
            ("How to access remote areas after floods?", "Access methods: 1) Helicopter transport, 2) Boat navigation, 3) Walking on foot, 4) Temporary bridges, 5) Alternative routes, 6) Community networks."),
            ("What infrastructure is most vulnerable?", "Most vulnerable: 1) Unpaved roads, 2) Wooden bridges, 3) Mud-brick buildings, 4) Open wells, 5) Pit latrines, 6) Temporary structures.")
        ]
        
        for question, answer in infrastructure_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:3]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 7. Agriculture Impacts (40 questions)
        agriculture_questions = [
            ("How do floods destroy crops like sorghum?", "Crop destruction: 1) Waterlogging kills plants, 2) Soil erosion, 3) Seed loss, 4) Delayed planting, 5) Reduced yields, 6) Food insecurity."),
            ("What happens to livestock during floods?", "Livestock impacts: 1) Drowning deaths, 2) Disease outbreaks, 3) Feed shortages, 4) Migration stress, 5) Market disruption, 6) Economic losses."),
            ("How do floods affect soil fertility?", "Soil impacts: 1) Nutrient leaching, 2) Erosion, 3) Sedimentation, 4) Compaction, 5) Salinity, 6) Reduced productivity."),
            ("How do floods disrupt planting seasons?", "Season disruption: 1) Delayed land preparation, 2) Seed unavailability, 3) Waterlogged fields, 4) Labor displacement, 5) Timing misalignment.")
        ]
        
        for question, answer in agriculture_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:4]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 8. Health Impacts (40 questions)
        health_questions = [
            ("What waterborne diseases occur after floods?", "Waterborne diseases: 1) Cholera outbreaks, 2) Diarrheal diseases, 3) Typhoid fever, 4) Hepatitis A, 5) Skin infections, 6) Eye infections."),
            ("How do floods affect healthcare access?", "Healthcare access: 1) Clinic closures, 2) Staff displacement, 3) Medicine shortages, 4) Transport barriers, 5) Equipment damage."),
            ("How do floods cause malnutrition?", "Malnutrition causes: 1) Food crop loss, 2) Market disruption, 3) Income loss, 4) Displacement, 5) Feeding program interruption."),
            ("What are the mental health effects?", "Mental health impacts: 1) Trauma from loss, 2) Displacement stress, 3) Anxiety, 4) Depression, 5) PTSD, 6) Community breakdown.")
        ]
        
        for question, answer in health_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:4]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 9. Social & Economic Effects (40 questions)
        social_questions = [
            ("How do floods cause school dropouts?", "School dropout causes: 1) School destruction, 2) Family displacement, 3) Economic hardship, 4) Child labor needs, 5) Long-term closure."),
            ("How do floods affect employment?", "Employment impacts: 1) Job losses, 2) Business closures, 3) Agricultural disruption, 4) Market collapse, 5) Income reduction."),
            ("What happens to housing after floods?", "Housing impacts: 1) Structure collapse, 2) Mud damage, 3) Mold growth, 4) Displacement, 5) Reconstruction needs."),
            ("How do floods create poverty cycles?", "Poverty cycles: 1) Asset loss, 2) Debt accumulation, 3) Reduced income, 4) Limited recovery, 5) Vulnerability increase.")
        ]
        
        for question, answer in social_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:4]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # 10. Humanitarian Response (40 questions)
        humanitarian_questions = [
            ("How do UN and NGOs coordinate relief?", "Relief coordination: 1) Cluster system, 2) Joint assessments, 3) Resource sharing, 4) Avoiding duplication, 5) Government partnership."),
            ("Where are emergency shelters located?", "Shelter locations: 1) Schools, 2) Churches, 3) Community centers, 4) Higher ground areas, 5) Government buildings."),
            ("What are the logistics challenges?", "Logistics challenges: 1) Road access, 2) Fuel shortages, 3) Security risks, 4) Remote locations, 5) Weather conditions."),
            ("What recovery programs exist?", "Recovery programs: 1) Livelihood support, 2) Infrastructure rebuilding, 3) Capacity building, 4) Disaster risk reduction, 5) Resilience building.")
        ]
        
        for question, answer in humanitarian_questions:
            qa_pairs.append({"question": question, "answer": answer})
            variations = generate_question_variations(question)
            for variation in variations[:4]:
                qa_pairs.append({"question": variation, "answer": answer})
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        logger.info(f"Saved dataset to {output_path}")
        
        return qa_pairs
    
    except Exception as e:
        logger.error(f"Error generating dataset: {str(e)}")
        raise

if __name__ == "__main__":
    create_qa_dataset()