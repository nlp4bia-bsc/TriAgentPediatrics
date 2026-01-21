"""
Triage Aggregator Agent
Validates individual specialty results, resolves conflicts, and produces final triage decision.
"""

from typing import Dict, List, Any, Tuple
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings

from core.models import PatientContext, TriageLevel, SpecialtyValidation, FinalTriageDecision, SpecialtyType, SpecialtyTriageResult


# ============================================================================
# Aggregation System Prompt
# ============================================================================

AGGREGATION_SYSTEM_PROMPT = f"""You are a medical triage aggregation expert. Your role is to synthesize multiple specialty triage results into a single, clinically coherent final decision.

## Your Task
Given validated results from multiple medical specialties and the triage guidelines they are based on, determine:
1. Which triage level should take precedence
2. How to resolve conflicts between specialties
3. Whether the case requires human review
4. Whether the case falls outside guideline coverage, and if so, a detailed explanation of the triage decision


##<total_guidelines>{open("guidelines/full_guidelines.txt", 'r').read()}</total_guidelines>

## Aggregation Principles

### Safety-First Hierarchy
- URGENT > MODERATE > MILD
- If ANY specialty suggests URGENT, strongly consider URGENT unless clearly invalid
- Exception: Invalid urgency escalations should be corrected

### Conflict Resolution Strategies

**Urgency Mismatch** (e.g., Fever says URGENT, Neurological says MILD):
- Identify which specialty is driving urgency
- Check if urgent specialty's reasoning is valid; if so, prioritize it

**Contradictory Findings** (e.g., "fever present" vs "no fever"):
- This indicates data extraction error or ambiguous case
- Flag for human review in case of doubt
- Use most conservative interpretation

**Overlapping Symptoms** (e.g., both Fever and Respiratory claim same symptom):
- Identify the primary specialty for the symptom
- Use clinical context to determine which interpretation is most relevant
- Document the relationship between specialties

**Interaction between Specialties** (e.g., Fever findings may impact Gastrointestinal urgency):
- Consider how findings in one specialty may influence another
- Adjust triage levels accordingly, prioritizing the most severe interpretation

### Clinical Coherence
Assess overall coherence (0-1 score):
- **High (0.8-1.0)**: All findings align, clear diagnosis pathway
- **Medium (0.5-0.8)**: Mostly coherent with minor inconsistencies
- **Low (0.0-0.5)**: Contradictions, atypical presentation, unclear picture

### Human Review Triggers
Flag for human review when:
- Multiple INVALID specialty results
- Unresolvable conflicts between specialties
- Case falls UNGUIDELINE in multiple specialties
- Clinical coherence score < 0.6
- Unexpected specialty combinations (rare disease patterns)

### Guideline Coverage Assessment
Case falls outside guidelines when:
- Novel symptom combinations not in the given guideline
- Atypical patient condition or co-morbiditiess
- Missing critical information to apply guidelines

## Output Requirements
Provide comprehensive `FinalTriageDecision` including:
- Final triage level with clear justification
- All specialty validations
- Conflict resolutions (if any)
- Human review flags and reasons
- Guideline gap identification
- Clinical coherence assessment

IMPORTANT: When filling out reasoning fields, BE CONCISE.
Limit specific reasoning fields to 1-2 sentences.
Do not repeat information already present in other fields
"""


# ============================================================================
# Triage Aggregator
# ============================================================================

class TriageAggregator:
    """
    Aggregates multiple specialty triage results into final decision.
    Uses AI to validate rule-based results and resolve conflicts.
    """
    
    def __init__(self, hf_model: OutlinesModel):
        """
        Initialize aggregator with AI model.
        
        Args:
            hf_model: OutlinesModel for validation and aggregation
        """
        
        # Aggregation agent - synthesizes all results
        self.aggregation_agent = Agent(
            model=hf_model,
            output_type=FinalTriageDecision,
            system_prompt=AGGREGATION_SYSTEM_PROMPT,
            retries=2,
        )
    
    async def aggregate(
        self,
        patient_ctx: PatientContext,
        specialty_validated_results: Dict[SpecialtyType, Dict[str, Any]]
    ) -> FinalTriageDecision:
        """
        Aggregate multiple specialty results into final triage decision.
        
        Args:
            patient_ctx: Original patient context
            specialty_validated_results: Results from each specialty evaluation
        
        Returns:
            FinalTriageDecision with validated, conflict-resolved result
        """
        # Filter out failed extractions
        valid_results = {
            specialty: results 
            for specialty, results in specialty_validated_results.items()
            if results['status'] == 'success'
        }
        
        if not valid_results:
            # There is something wrong with the extraction Agent. Do not continue
            return self._all_failed_result()
        
        # Step 2: Aggregate validated results into final decision
        final_decision = await self._aggregate_validations(
            patient_ctx,
            valid_results
        )
        
        return final_decision
    
    async def _aggregate_validations(
        self,
        patient_ctx: PatientContext,
        specialty_validated_results: Dict[SpecialtyType, Dict[str, Any]]
    ) -> FinalTriageDecision:
        """
        Synthesize all validated results into final decision.
        """
        prompt = self._build_aggregation_prompt(
            patient_ctx,
            specialty_validated_results
        )
        
        final_result = await self.aggregation_agent.run(
            prompt,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 600})
        )
        
        return final_result.output
    
    def _build_aggregation_prompt(
        self,
        patient_ctx: PatientContext,
        specialty_validated_results: Dict[SpecialtyType, Dict[str, Any]]
    ) -> str:
        """Build prompt for aggregating all results."""
        prompt_parts = [
            "Aggregate these specialty triage results into a final decision:\n",
            f"\n<patient_case>\n{patient_ctx._patient_summary}\n</patient_case>",
            "\n<specialty_triage>\n"
        ]
        
        summary = []
        for specialty, results in specialty_validated_results.items():
            rb_triage: SpecialtyTriageResult = results['triage']
            validation: SpecialtyValidation = results['validation']
            prompt_parts.append(
                f"\n### Specialty: {specialty.value}\n"
                f"<{specialty.value}_rule_based_triage>\n" + rb_triage.format_for_explanation() + f"\n</{specialty.value}_rule_based_triage>\n"
                f"<{specialty.value}_validation>\n" + validation.format_for_explanation() + f"\n</{specialty.value}_validation>\n"
            )
            summary.append(
                f"- {specialty.value} triage: {validation.validated_level.value} | Reasoning: {validation.validation_reasoning}"
            )
        
        prompt_parts.append("\n</specialty_triage>\n")
        prompt_parts.append("\n### Summary of Specialty Validations:\n")
        prompt_parts.append("\n".join(summary))

        prompt_parts.append(
            "\n\nAGGREGATION TASK:"
            "\n1. Determine final triage level (consider safety-first principle)"
            "\n2. Identify and resolve any conflicts between specialties"
            "\n3. Assess whether human review is needed"
            "\n4. Identify any guideline gaps or edge cases"
            "\n5. Evaluate overall clinical coherence (0-1 score)"
            "\n\nProvide comprehensive final triage decision."
        )
        
        return "\n".join(prompt_parts)
    
    def _all_failed_result(self) -> FinalTriageDecision:
        """Return result when all specialty extractions failed."""
        return FinalTriageDecision(
            final_level=TriageLevel.UNMATCHED,
            conflicts_detected=[],
            reasoning="All specialty data extractions failed",
            requires_human_review=True,
            review_reasons=["Complete extraction failure"],
            falls_outside_guidelines=True,
            guideline_gaps=["Unable to extract any clinical information"],
            clinical_coherence_score=0.0,
            coherence_notes="No valid data to assess coherence"
        )