"""
Safety Checker for Medical Triage
Implements hard-coded safety overrides and AI-assisted red flag detection.
"""

from enum import Enum
from typing import List, Optional, Set
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings

from core.models import PatientContext, SafetyAssessment, RedFlagCategory

SAFETY_ASSESSMENT_PROMPT = """You are a medical safety assessment agent. Your job is to identify RED FLAG symptoms that require immediate emergency care.

## Your Task
Analyze the patient case for life-threatening or time-critical conditions that require immediate Class 1 (Urgent) triage classification, regardless of other factors.

## Red Flag Categories to Detect

1. **AIRWAY/BREATHING**: Severe respiratory distress, airway obstruction, inability to breathe
2. **CIRCULATION**: Uncontrolled bleeding, shock, severe hypotension
3. **CONSCIOUSNESS**: Unresponsive, altered mental status, confusion, lethargy
4. **SEVERE PAIN**: Worst pain of life, thunderclap headache, crushing chest pain
5. **PEDIATRIC EMERGENCY**: Infant not feeding, inconsolable, non-blanching rash, petechiae
6. **TRAUMA**: Major trauma, loss of consciousness, penetrating injuries
7. **CARDIAC**: Chest pain with cardiac features, symptoms of heart attack
8. **NEUROLOGICAL**: Stroke symptoms, ongoing seizure, sudden severe headache
9. **SEPSIS**: Signs of severe infection with shock
10. **HEMORRHAGE**: Massive GI bleeding, hematemesis, uncontrolled hemorrhage

## Critical Assessment Rules

### MUST Escalate (Automatic Class 1):
- Any mention of "unresponsive", "unconscious", "not breathing"
- Active uncontrolled bleeding or hemorrhage
- Chest pain with cardiac features (radiating, diaphoresis, crushing)
- Stroke symptoms (facial droop, slurred speech, weakness)
- Severe respiratory distress or airway compromise
- Shock or severe hypotension
- Severe trauma with loss of consciousness

## Output Requirements

You must return:
- `requires_immediate_escalation`: true/false
- `detected_red_flags`: List of red flag categories found (use category names exactly)
- `reasoning`: Clear explanation of which symptoms triggered escalation

## Examples

Example 1:
Input: "13-year-old male with crushing chest pain radiating to left arm, sweating profusely, started 30 minutes ago"
Output:
{
  "requires_immediate_escalation": true,
  "detected_red_flags": ["cardiac", "severe_pain"],
  "reasoning": "Classic presentation of acute coronary syndrome: crushing chest pain with radiation to arm and diaphoresis. This is a time-critical cardiac emergency requiring immediate intervention.",
}

Example 2:
Input: "15-year-old with sudden severe headache described as 'worst of my life', started suddenly while exercising"
Output:
{
  "requires_immediate_escalation": true,
  "detected_red_flags": ["neurological", "severe_pain"],
  "reasoning": "Thunderclap headache (worst headache of life with sudden onset) is a red flag for subarachnoid hemorrhage or other serious intracranial pathology. Requires immediate imaging and evaluation.",
}

Example 3:
Input: "8-year-old with mild cough and runny nose for 2 days"
Output:
{
  "requires_immediate_escalation": false,
  "detected_red_flags": [],
  "reasoning": "No red flag symptoms identified. Upper respiratory symptoms without alarm features do not require immediate escalation.",
}

## Important Notes
- Don't be too conservative: If uncertain whether something is a red flag, don't escalate it since it's urgency will be better assessed in further, more in-depth analysis
- Look for combinations: Multiple concerning symptoms together may create a red flag
- Consider the complete picture: Age + symptoms + duration
- Never miss: False positives are acceptable; false negatives are not
"""


# ============================================================================
# Safety Checker Implementation
# ============================================================================

class SafetyChecker:
    
    def __init__(self, hf_model: Optional[OutlinesModel] = None):
        """
        Initialize safety checker.
        
        Args:
            hf_model: Optional AI model for intelligent red flag detection.
                     If None, only rule-based checking is performed.
        """
        self.hf_model = hf_model
        
        if hf_model is not None:
            self.agent = Agent(
                model=hf_model,
                output_type=SafetyAssessment,
                system_prompt=SAFETY_ASSESSMENT_PROMPT,
                retries=2,
            )
        else:
            self.agent = None
    
    async def check_immediate_escalation(self, patient_ctx: PatientContext) -> SafetyAssessment:
        """
        Use AI model to detect complex or subtle red flags.
        """
        if self.agent is None:
            return SafetyAssessment(
                requires_immediate_escalation=False,
                detected_red_flags=[],
                reasoning="AI-assisted safety checking not available.",
            )
        
        prompt = patient_ctx._patient_summary
        
        result = await self.agent.run(
            prompt,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 300})
        )
        
        return result.output