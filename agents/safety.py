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

SAFETY_ASSESSMENT_PROMPT = """You are a SAFETY OVERRIDE MODULE, not a diagnostic system.

Your ONLY job is to detect EXTREME, IMMEDIATE, LIFE-THREATENING RED FLAGS
that require emergency escalation within MINUTES TO HOURS.

If a patient can reasonably wait for same-day or routine medical evaluation,
you MUST return requires_immediate_escalation = false.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 ABSOLUTE PROHIBITIONS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT:
- Suggest or imply diagnoses (e.g. Kawasaki, UTI, vasculitis, sepsis)
- Escalate based on "possibility", "risk", or "needs evaluation"
- Escalate based on age alone (infant ≠ emergency)
- Escalate mild/moderate symptoms without instability
- Escalate because something is “concerning” or “should be checked”
- Perform early diagnosis or clinical reasoning beyond red flags

If escalation is based on speculation, YOU ARE WRONG.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ WHEN TO ESCALATE (VERY STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Escalate ONLY if there is CLEAR evidence of:

1. AIRWAY/BREATHING  
   - Inability to breathe
   - Severe respiratory distress
   - Cyanosis, gasping, apnea

2. CIRCULATION  
   - Uncontrolled active bleeding
   - Signs of shock (collapse, mottled skin, weak pulse)
   - Severe hypotension with symptoms

3. CONSCIOUSNESS  
   - Unresponsive
   - Confusion, coma, sudden altered mental status
   - Active seizure or post-ictal unresponsiveness

4. SEVERE PAIN  
   - “Worst pain of life”
   - Thunderclap headache
   - Pain causing collapse or inability to function

5. PSYCHOLOGICAL  
   - Suicide attempt
   - Explicit suicidal intent or plan

6. TRAUMA  
   - Major trauma
   - Head injury with loss of consciousness
   - Penetrating injuries

7. CARDIAC  
   - Crushing chest pain with cardiac features
   - Syncope with chest pain or palpitations

8. NEUROLOGICAL  
   - Stroke symptoms (face droop, speech loss, paralysis)
   - Ongoing seizure

9. HEMORRHAGE (VERY NARROW)
   - Massive bleeding
   - Hematemesis
   - Large-volume GI bleeding
   - Bleeding causing hemodynamic instability

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 EXPLICIT NON-RED FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These MUST NOT trigger escalation by themselves:

- Fever (any value) without shock or altered consciousness
- Rash without hypotension or airway compromise
- Blood in urine without hemodynamic instability
- Mild hematochezia without signs of shock
- Pain with urination
- Diarrhea with blood but no systemic instability
- Lethargy without altered consciousness
- “Could be serious if untreated”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 DECISION RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask yourself ONE question:

"If this patient is NOT escalated immediately,
is there a realistic risk of death or irreversible harm
within the next few hours?"

If the answer is NO → requires_immediate_escalation = false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return:
- requires_immediate_escalation: true/false
- detected_red_flags: list of categories (ONLY if true)
- reasoning: factual description of the EXACT red flag found

If no explicit red flag is present:
- requires_immediate_escalation MUST be false
- detected_red_flags MUST be empty

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