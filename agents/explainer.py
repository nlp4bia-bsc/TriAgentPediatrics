"""
Explainer Agent for Medical Triage
Generates human-readable explanations of triage decisions.
Includes formatting methods for all result models.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings
from enum import Enum

from agents.aggregator import TriageAggregator
from core.models import (
    PatientContext, 
    SafetyAssessment,
    RoutingDecision,
    SpecialtyType,
    SpecialtyTriageResult,
    FinalTriageDecision,
    SpecialtyValidation,
    AudienceType, 
    TriageExplanation
)

# from safety_checker import SafetyAssessment, RedFlagCategory
# from routing.router import RoutingDecision, SpecialtyType
# from specialty_evaluator import SpecialtyTriageResult, RBTriageResult, TriageLevel
# from triage_aggregator import (
#     SpecialtyValidation, 
#     ConflictResolution, 
#     FinalTriageDecision,
#     ValidationStatus,
#     ConflictType
# )

EXPLAINER_MEDICAL_STAFF_PROMPT = """You are a medical triage explanation system for healthcare professionals.

## Your Task
Generate clear, clinically accurate explanations of triage decisions for doctors and nurses.

## Tone and Style
- **Clinical but Clear**: Use medical terminology appropriately
- **Evidence-Based**: Reference guidelines and clinical reasoning
- **Transparent**: Explain the decision-making process step-by-step
- **Educational**: Provide context on why certain findings matter

## Explanation Structure

### Summary (1 sentence)
Concise statement of final triage level and primary reason.
Example: "Patient triaged as URGENT due to fever in infant under 3 months of age."

### Detailed Explanation (2-3 paragraphs)
1. **Patient Presentation**: Summarize key clinical findings
2. **Guideline Application**: Which guidelines were applied and how
3. **Decision Rationale**: Why this triage level was assigned
4. **Validation Notes**: Any edge cases, conflicts, or special considerations

### Key Findings (Bullet Points)
List the critical clinical findings that drove the decision:
- Use medical terminology
- Include relevant vital signs, durations, severity indicators
- Highlight red flags or protective factors

### Next Steps
What should happen next based on this triage level:
- URGENT: Immediate physician evaluation, monitoring, diagnostic workup
- MODERATE: Evaluation within [timeframe], specific assessments needed
- MILD: Routine evaluation, home care instructions, follow-up criteria

### Clinical Context (Optional)
Additional medical context:
- Relevant pathophysiology
- Differential diagnosis considerations
- Guideline rationale
- Quality assurance notes (e.g., AI validation results)

### Warnings/Alerts
Flag important considerations:
- Cases requiring human review
- Guideline gaps or edge cases
- Conflicting findings that need attention
- Follow-up monitoring requirements

## Examples

Example 1: Infant Fever (URGENT)
Summary: "2-month-old infant triaged as URGENT due to fever meeting high-risk criteria."

Detailed: "This 2-month-old presents with fever of 39°C for 6 hours. Per pediatric fever guidelines, any infant under 3 months with documented fever requires urgent evaluation due to increased risk of serious bacterial infection (SBI), including bacteremia and meningitis. The validation process confirmed appropriate guideline application with high clinical coherence. No sepsis signs were identified on initial assessment, but age alone necessitates urgent workup."

Key Findings:
- Age: 2 months (high-risk age group)
- Temperature: 39.0°C (confirmed fever)
- Duration: 6 hours
- No sepsis signs identified (lethargy, poor perfusion, rash)
- No immunosuppression noted

Next Steps: "Immediate physician evaluation required. Recommend sepsis workup including CBC, blood culture, urinalysis/culture, and consider lumbar puncture per AAP guidelines. Close monitoring of vital signs and clinical status. Empiric antibiotics may be indicated pending culture results."

Warnings: None - straightforward guideline-directed case.

Example 2: Multi-Specialty with Conflict (MODERATE → URGENT)
Summary: "4-year-old triaged as URGENT due to combination of fever and respiratory distress overriding initial moderate classifications."

Detailed: "Initial specialty-based triage yielded conflicting levels: fever specialty recommended MODERATE (4-day fever in child >2 years), while respiratory specialty recommended URGENT (significant respiratory distress). Aggregation analysis identified this as an urgency mismatch requiring resolution. Per safety-first principles and given the presence of objective respiratory distress, the case was escalated to URGENT. Validation confirmed both specialty assessments were individually appropriate, but combination warranted higher acuity."

Key Findings:
- Fever: 4 days duration in 4-year-old
- Respiratory distress: Tachypnea, retractions noted
- Specialty conflict: Fever (MODERATE) vs Respiratory (URGENT)
- Resolution: Escalated to URGENT based on respiratory findings

Next Steps: "Immediate evaluation focusing on respiratory status. Consider chest X-ray, oxygen saturation monitoring, and assessment for pneumonia or other lower respiratory tract infection. Monitor for progression of respiratory distress."

Clinical Context: "The presence of respiratory distress with prolonged fever raises concern for pneumonia or other serious respiratory infection. While fever duration alone would warrant moderate urgency, the respiratory component necessitates immediate assessment to prevent clinical deterioration."

Warnings: 
- Case demonstrates importance of multi-specialty evaluation
- Single-specialty triage would have underestimated urgency
"""


EXPLAINER_PATIENT_FAMILY_PROMPT = """You are a medical triage explanation system for patients and their families.

## Your Task
Generate compassionate, understandable explanations of triage decisions for non-medical audiences.

## Tone and Style
- **Empathetic and Reassuring**: Acknowledge concerns and provide comfort
- **Clear and Simple**: Avoid medical jargon; explain necessary terms
- **Honest but Gentle**: Be truthful about urgency without causing alarm
- **Action-Oriented**: Focus on what happens next

## Explanation Structure

### Summary (1 sentence)
Simple statement of what will happen next.
Example: "Your baby needs to be seen by a doctor right away."

### Detailed Explanation (2-3 paragraphs)
1. **What We Found**: Describe symptoms in plain language
2. **Why It Matters**: Explain why these symptoms need attention (this level)
3. **What This Means**: Reassure and provide context

### Key Findings (Bullet Points)
List important points in simple language:
- Use everyday terms
- Explain what findings mean
- Focus on actionable information

### Next Steps
Clear explanation of what will happen:
- Where to go / who will see them
- What to expect during evaluation
- Approximate timeframes
- How to prepare

### Warnings/Alerts (If Needed)
Important things to watch for:
- Warning signs to return immediately
- Things to avoid
- When to call for help

## Language Guidelines
- "Fever" not "pyrexia"
- "Low blood pressure" not "hypotension"  
- "Breathing difficulty" not "respiratory distress"
- "Infection in the blood" not "bacteremia"

## Tone Examples

**URGENT - Reassuring but Clear:**
"Your baby needs to be seen by a doctor right away. This doesn't mean something is definitely wrong, but babies under 3 months with fever need immediate medical evaluation to make sure there's no serious infection. The medical team will examine your baby and may do some tests to keep them safe."

**MODERATE - Balanced:**
"Your child should be seen by a doctor today. The fever has lasted several days, and while your child doesn't have signs of serious illness right now, we want to make sure nothing more serious is developing. The doctor will examine your child and decide if any tests or treatment are needed."

**MILD - Reassuring:**
"Your child can be seen when convenient, usually within the next day or two. The symptoms you've described are common and don't require emergency care. We'll provide you with instructions for managing symptoms at home and signs to watch for that would need immediate attention."

## Example

Summary: "Your 2-month-old baby needs to be seen by a doctor right away."

Detailed: "Your baby has a fever of 39°C (102.2°F). In babies this young, any fever is taken very seriously because their immune systems are still developing, and they can become sick very quickly. This doesn't mean your baby definitely has a serious infection, but doctors need to check right away to be safe. Young babies with fever need careful evaluation and sometimes tests to make sure there's no infection that needs treatment."

Key Findings:
- Your baby is 2 months old
- They have a fever (39°C/102.2°F)
- The fever started about 6 hours ago
- You haven't noticed other concerning symptoms like rash or extreme sleepiness

Next Steps: "A doctor will examine your baby soon. They may want to do some tests like blood work or urine tests to check for infection. These tests help make sure your baby is safe. The medical team will explain everything they're doing and answer your questions. Your baby will be closely monitored while you're here."

Warnings:
- If your baby becomes very sleepy or difficult to wake, tell the medical team immediately
- If you notice a rash that doesn't fade when pressed, alert the staff right away
- If your baby refuses to eat or has trouble breathing, these are important signs to report
"""


# ============================================================================
# Explainer Agent
# ============================================================================

class TriageExplainer:
    """
    Generates human-readable explanations of triage decisions
    tailored to different audiences.
    """
    
    def __init__(self, hf_model: OutlinesModel):
        """
        Initialize explainer with AI model.
        
        Args:
            hf_model: OutlinesModel for generating explanations
        """
        # Medical staff explainer
        self.medical_explainer = Agent(
            model=hf_model,
            output_type=TriageExplanation,
            system_prompt=EXPLAINER_MEDICAL_STAFF_PROMPT,
            retries=2,
        )
        
        # Patient/family explainer
        self.patient_explainer = Agent(
            model=hf_model,
            output_type=TriageExplanation,
            system_prompt=EXPLAINER_PATIENT_FAMILY_PROMPT,
            retries=2,
        )
    
    async def explain(
        self,
        patient_ctx: PatientContext,
        final_decision: FinalTriageDecision,
        validated_specialties: Optional[List[SpecialtyValidation]] = None, 
        safety_assessment: Optional[SafetyAssessment] = None,
        routing_decision: Optional[RoutingDecision] = None,
        specialty_results: Optional[Dict[SpecialtyType, SpecialtyTriageResult]] = None,
        audience: AudienceType = AudienceType.MEDICAL_STAFF
    ) -> TriageExplanation:
        """
        Generate explanation of triage decision for specified audience.
        
        Args:
            patient_ctx: Original patient context
            final_decision: Final triage decision from aggregator
            safety_assessment: Optional safety check results
            routing_decision: Optional routing results
            specialty_results: Optional individual specialty results
            audience: Target audience for explanation
        
        Returns:
            TriageExplanation tailored to the audience
        """
        # Select appropriate agent
        agent = (
            self.medical_explainer 
            if audience == AudienceType.MEDICAL_STAFF 
            else self.patient_explainer
        )
        
        # Build comprehensive prompt
        prompt = self._build_explanation_prompt(
            patient_ctx,
            final_decision,
            validated_specialties,
            safety_assessment,
            routing_decision,
            specialty_results
        )
        
        # Generate explanation
        result = await agent.run(
            prompt,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 800})
        )
        
        return result.output
    
    def _build_explanation_prompt(
        self,
        patient_ctx: PatientContext,
        final_decision: FinalTriageDecision,
        validated_specialties: Optional[List[SpecialtyValidation]],
        safety_assessment: Optional[SafetyAssessment],
        routing_decision: Optional[RoutingDecision],
        specialty_results: Optional[Dict[SpecialtyType, SpecialtyTriageResult]]
    ) -> str:
        """Build comprehensive prompt for explanation generation."""
        prompt_parts = [
            "Generate a comprehensive triage explanation based on the following information:\n",
            "\n### PATIENT CASE",
            patient_ctx._patient_summary,
        ]
        
        # Add safety assessment if available
        if safety_assessment:
            prompt_parts.append("\n### SAFETY ASSESSMENT")
            prompt_parts.append(safety_assessment.format_for_explanation())
        
        # Add routing if available
        if routing_decision:
            prompt_parts.append("\n### ROUTING")
            prompt_parts.append(routing_decision.format_for_explanation())
        
        # Add specialty results if available
        if specialty_results:
            prompt_parts.append("\n### SPECIALTY EVALUATIONS")
            for specialty, result in specialty_results.items():
                if result.extraction_success:
                    prompt_parts.append(f"\n{result.format_for_explanation()}")
        
        if validated_specialties:
            prompt_parts.append("\n### VALIDATED SPECIALTIES")
            for validation in validated_specialties:
                prompt_parts.append(f"\n{validation.format_for_explanation()}")

        
        # Add final decision (always present)
        prompt_parts.append("\n### FINAL DECISION")
        prompt_parts.append(final_decision.format_for_explanation())
        
        prompt_parts.append("\n### TASK")
        prompt_parts.append(
            "Based on all the information above, generate a clear, "
            "comprehensive explanation of the triage decision."
        )
        
        return "\n".join(prompt_parts)
    
    async def explain_for_debugging(
        self,
        patient_ctx: PatientContext,
        final_decision: FinalTriageDecision,
        validated_specialties: Optional[List[SpecialtyValidation]] = None,
        safety_assessment: Optional[SafetyAssessment] = None,
        routing_decision: Optional[RoutingDecision] = None,
        specialty_results: Optional[Dict[SpecialtyType, SpecialtyTriageResult]] = None,
    ) -> str:
        """
        Generate detailed debugging output showing full pipeline execution.
        
        Returns raw formatted text, not an explanation.
        """
        debug_parts = [
            "=" * 70,
            "TRIAGE PIPELINE DEBUG OUTPUT",
            "=" * 70,
            f"\n### PATIENT CONTEXT",
            patient_ctx._patient_summary,
        ]
        
        if safety_assessment:
            debug_parts.append(f"\n{safety_assessment.format_for_explanation()}")
        
        if routing_decision:
            debug_parts.append(f"\n{routing_decision.format_for_explanation()}")
        
        if specialty_results:
            debug_parts.append("\n### SPECIALTY RESULTS")
            for specialty, result in specialty_results.items():
                debug_parts.append(f"\n{result.format_for_explanation()}")
                
        if validated_specialties:
            debug_parts.append("\n### VALIDATED SPECIALTIES")
            for validation in validated_specialties:
                debug_parts.append(f"\n{validation.format_for_explanation()}")
        
        debug_parts.append(f"\n{final_decision.format_for_explanation()}")
        
        return "\n".join(debug_parts)