"""
Triage Router Agent using Pydantic AI
Routes patient cases to relevant medical specialties for evaluation.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.models import PatientContext, SpecialtyType, RoutingDecision
import torch


# ============================================================================
# Domain Models
# ============================================================================




# ============================================================================
# System Prompt
# ============================================================================

ROUTING_SYSTEM_PROMPT = """You are a medical triage routing agent. Your job is to analyze a patient case and determine which medical specialty modules should evaluate the patient.

## Your Task
Given a patient's chief complaint, age, and sex, identify ALL relevant medical specialties that should perform triage evaluation.

## Guidelines
1. **Be Comprehensive**: Include all specialties that might be relevant. It's better to over-include than miss a critical specialty.
   
2. **Specialty Routing Descriptions**:
    * **FEVER:** Any mention of elevated temperature, chills, feverishness, or thermometry readings above normal. Includes persistent fever or fever in infants/vulnerable patients.
    * **RESPIRATORY:** Breathing difficulties, shortness of breath, noisy breathing (stridor/wheezing), cough, choking, or signs of low oxygen (blue/gray skin).
    * **NEUROLOGICAL:** Seizures, fainting, loss of consciousness, confusion, extreme irritability, sudden weakness, speech difficulty, or severe headaches.
    * **TRAUMA:** Physical injuries from accidents, falls, or hits. Includes suspected fractures, deep cuts, heavy bleeding, burns, and head injuries.
    * **GASTROINTESTINAL:** Abdominal/stomach pain, vomiting (including projectile), diarrhea, dehydration, blood in stool, or suspected appendicitis.
    * **GENITOURINARY:** Changes in urine color (dark/bloody), pain during urination, testicular pain, ovarian pain, or menstrual irregularities.
    * **CARDIOVASCULAR:** Chest pain, heart racing (tachycardia), slow heart rate, fainting with low blood pressure, or extreme paleness with cold skin.
    * **OTORHINOLARYNGOLOGY:** Earaches, ear discharge, nosebleeds, foreign objects in the nose or ear, sore throat, or dental/mouth pain.
    * **OPHTHALMOLOGY:** Red eyes, eye pain, eyelid swelling, or any sudden changes/disturbances in vision.
    * **DERMATOLOGY:** Skin rashes, hives, acne, warts, skin infections, or minor superficial skin changes/growths.
    * **PSYCHIATRIC:** Mental health crises, suicidal thoughts, self-harm, aggression, hallucinations, or behavioral changes related to substance use.
    * **ALLERGY:** Allergic reactions, facial swelling, or hives—specifically distinguishing between skin-only reactions and those affecting breathing.
    * **TOXICOLOGY:** Swallowing non-food items (batteries, magnets), accidental or intentional medication overdose, or exposure to poisonous substances.
    * **ENDOCRINE:** Blood sugar issues (high or low glucose), complications of diabetes, or known metabolic disease crises.
    * **OTHER:** Administrative tasks (referrals/forms), vaccinations, check-ups, chronic disease monitoring, or general fatigue.

3. **Overlapping Symptoms**: Many symptoms can indicate multiple conditions:
   - Chest pain → CARDIOVASCULAR + RESPIRATORY
   - Abdominal pain with fever → GASTROINTESTINAL + FEVER
   - Headache with confusion → NEUROLOGICAL + possibly FEVER

## Output Format
You must return a JSON object with:
- `specialties`: List of specialty names that should evaluate this case
- `reasoning`: Clear explanation of why each specialty was selected

## Examples

Example 1:
Input: "3-month-old with fever of 39°C for 6 hours"
Output: 
{
  "specialties": ["fever"],
  "reasoning": "Infant under 3 months with fever requires urgent fever specialty evaluation due to high risk of serious bacterial infection."
}

Example 2:
Input: "45-year-old with chest pain and shortness of breath for 2 hours"
Output:
{
  "specialties": ["cardiovascular", "respiratory"],
  "reasoning": "Chest pain in adult requires cardiovascular evaluation for cardiac causes. Concurrent shortness of breath requires respiratory evaluation. Both are urgent symptoms."
}

Example 3:
Input: "8-year-old with vomiting, diarrhea for 2 days, but without fever"
Output:
{
  "specialties": ["gastrointestinal"],
  "reasoning": "Gastrointestinal symptoms (vomiting, diarrhea) require GI evaluation for dehydration and infectious gastroenteritis persisting 2 days."
}

Remember: When in doubt, include the specialty. Missing a relevant specialty could delay critical care.
"""

# ============================================================================
# Router Agent
# ============================================================================

class TriageRouter:
    """
    Medical triage router that determines which specialty modules 
    should evaluate a patient case.
    """
    
    def __init__(self, hf_model: OutlinesModel):
        """
        Initialize the router with a specific model.
        
        Args:
            hf_model: Initialized huggingface model
        """
        self.agent = Agent(
            model=hf_model,
            output_type=RoutingDecision,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            retries=2,
        )
    
    async def route(
        self, 
        patient_ctx: PatientContext
    ) -> RoutingDecision:
        """
        Route a patient case to relevant medical specialties.
        
        Args:
            patient_case: The patient case to route
        
        Returns:
            RoutingDecision with selected specialties and reasoning
        
        Raises:
            Exception: If routing fails after retries
        """
        # Build the prompt for the agent
        prompt = patient_ctx._patient_summary
        
        # Run the agent
        result = await self.agent.run(
            prompt,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 200})
        )
        
        return result.output