from typing import Dict, List, Any, Optional, Union
from enum import Enum, IntEnum
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.outlines import OutlinesModel

from core.models import TriageLevel, SpecialtyType, SpecialtyValidation, ValidationAction, FinalTriageDecision

# ============================================================================
# 1. THE ARBITER (Deterministic Logic)
# Factor 6: Processes - Business logic should be code, not prompts.
# ============================================================================

class AggregationState(BaseModel):
    """The deterministic state computed BEFORE the LLM runs."""
    proposed_level: TriageLevel
    most_urgent_specialty: SpecialtyType
    conflicting_specialties: List[SpecialtyType]
    is_critical_mismatch: bool  # True if levels differ by >1 step (e.g. 2 vs 5)
    has_unguideline_findings: bool
    base_rationale: str

class TriageArbiter:
    """
    Deterministic logic to enforce the Safety-First Hierarchy for string Enums.
    
    Hierarchy (High to Low Urgency):
    1. EMERGENCY_DEPARTMENT
    2. PRIMARY_CARE_TODAY
    3. PRIMARY_CARE_APPOINTMENT
    4. UNMATCHED (not handled since it is considered invalid if so)
    """

    # Define the weight of each level. Lower number = Higher Clinical Urgency.
    PRIORITY_MAP = {
        TriageLevel.EMERGENCY_DEPARTMENT: 1,
        TriageLevel.PRIMARY_CARE_TODAY: 2,
        TriageLevel.PRIMARY_CARE_APPOINTMENT: 3
    }

    @staticmethod
    def compute_state(valid_results: Dict[SpecialtyType, Dict[str, Any]]) -> AggregationState:
        # 1. Flatten and weight the data
        evaluations = []
        has_unguideline_findings = False
        
        for specialty, data in valid_results.items():
            val: SpecialtyValidation = data['validation']
            # Map the Enum level to a comparable weight
            weight = TriageArbiter.PRIORITY_MAP.get(val.validated_level, 99)
            
            evaluations.append({
                'specialty': specialty,
                'level': val.validated_level,
                'weight': weight,
                'action': val.validation_action
            })
            
            if val.validation_action == ValidationAction.UNGUIDELINE:
                has_unguideline_findings = True

        if not evaluations:
            return AggregationState(
                proposed_level=TriageLevel.UNMATCHED,
                most_urgent_specialty=SpecialtyType.FEVER, # Fallback
                conflicting_specialties=[],
                is_critical_mismatch=True,
                has_unguideline_findings=True,
                base_rationale="Extraction failure: No specialty data reached the arbiter."
            )

        # 2. Sort by Weight (Safety-First: Smallest weight wins)
        sorted_evals = sorted(evaluations, key=lambda x: x['weight'])
        
        highest_urgency = sorted_evals[0]
        lowest_urgency = sorted_evals[-1]
        
        # 3. Detect Conflicts
        # A 'Critical Mismatch' in this context is a gap of more than one destination.
        # e.g., ED vs Primary Care Appointment is critical. ED vs Primary Care Today is a standard conflict.
        weight_diff = abs(highest_urgency['weight'] - lowest_urgency['weight'])
        is_critical = weight_diff >= 2
        
        # Identify which specialties did not match the highest urgency found
        conflicts = [
            x['specialty'] for x in evaluations 
            if x['level'] != highest_urgency['level']
        ]
        
        rationale = (
            f"Safety Protocol: Prioritized {highest_urgency['level'].value} "
            f"due to {highest_urgency['specialty'].value} findings."
        )

        return AggregationState(
            proposed_level=highest_urgency['level'],
            most_urgent_specialty=highest_urgency['specialty'],
            conflicting_specialties=conflicts,
            is_critical_mismatch=is_critical,
            has_unguideline_findings=has_unguideline_findings,
            base_rationale=rationale
        )
# ============================================================================
# 2. THE GUARDIAN (The Agent)
# Factor 5: Build/Release/Run - Only tasked with semantic verification
# ============================================================================

GUARDIAN_SYSTEM_PROMPT = """You are a Clinical Logic Auditor (The Guardian). 
Your goal is NOT to guess the triage level—that has already been calculated by the Safety Protocol.
Your goal is to AUDIT that decision for physiological coherence and edge cases.

## Input Context
You will receive:
1. Patient Clinical Summary
2. Validated Specialty Opinions
3. The "Computed Safety State" (The proposed decision based on strict hierarchy)

## Your Responsibilities

1. **Verify Coherence (Physiological Check)**
   - Do the symptoms reported by Specialty A contradict Specialty B? (e.g., "High fever" vs "Hypothermia")
   - If yes, `clinical_coherence_score` should be low (< 0.5) and `requires_human_review` = True.

2. **Audit the Safety State**
   - The system defaults to the HIGHEST urgency found.
   - You act as a check: Does the union of symptoms actually support this high urgency, or is it a clear artifact?
   - *Note: Rarely overrule the Computed Safety State unless you see a hallucination.*

3. **Guideline Gap Detection**
   - If the combination of symptoms is complex (multisystemic), flag `guideline_gap_detected`.

## Output Rules
- Rationale fields MUST NOT exceed 2 sentences.
- Do not repeat information across fields.
- If `is_critical_mismatch` was flagged in input, you MUST explain why the specialties disagreed in `conflict_resolution_notes`.
"""

# ============================================================================
# 3. THE AGGREGATOR CLASS
# ============================================================================

class TriageAggregator:
    def __init__(self, out_model: OutlinesModel):
        self.arbiter = TriageArbiter()
        self.guardian_agent = Agent(
            model=out_model,
            output_type=FinalTriageDecision,
            system_prompt=GUARDIAN_SYSTEM_PROMPT,
            deps_type=AggregationState # Inject state into context
        )

    async def aggregate(
        self, 
        patient_ctx: Any, # Typed as PatientContext in your code
        specialty_validated_results: Dict[SpecialtyType, Dict[str, Any]]
    ) -> FinalTriageDecision:
        
        # --- PHASE 1: Dependency Check ---
        # Filter failures (Factor 2: Explicit Dependencies)
        valid_results = {
            s: r for s, r in specialty_validated_results.items() 
            if r.get('status') == 'success' and r.get('validated_level', None) != TriageLevel.UNMATCHED # if it is validated and matched to a triage level 
        }
        
        if not valid_results:
            return self._fail_safe_response("No valid extraction results.")

        # --- PHASE 2: The Arbiter (Deterministic) ---
        # Factor 6: Execute strict business logic before AI
        computed_state = self.arbiter.compute_state(valid_results)
        
        # --- PHASE 3: The Guardian (AI Reasoning) ---
        # Factor 8: Concurrency - We construct the view for the AI
        prompt = self._build_prompt(patient_ctx, valid_results, computed_state)
        
        result = await self.guardian_agent.run(
            prompt,
            deps=computed_state,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 600})
        )
        
        # --- PHASE 4: The Circuit Breaker (Safety) ---
        # Factor 9: Disposability - If AI returns low coherence, we force Human Review
        final_output = result.output
        if computed_state.is_critical_mismatch:
            final_output.requires_human_review = True
            if "Critical Mismatch" not in final_output.review_triggers:
                final_output.review_triggers.append(f"Critical mismatch in specialty findings ({len(computed_state.conflicting_specialties)} conflicts)")
        
        return final_output

    def _build_prompt(self, patient, results, state: AggregationState) -> str:
        """Constructs the audit dossier for the LLM."""
        lines = [
            f"# Patient Summary\n{getattr(patient, '_patient_summary', 'N/A')}\n",
            "# Computed Safety State (ARBITER DECISION)",
            f"PROPOSED LEVEL: {state.proposed_level.name} ({state.proposed_level.value})",
            f"RATIONALE: {state.base_rationale}",
            f"CRITICAL CONFLICT: {state.is_critical_mismatch}",
            "\n# Specialty Evidence"
        ]
        
        for spec, data in results.items():
            val = data['validation']
            lines.append(f"## {spec.value.upper()}")
            lines.append(val.format_for_explanation())
            lines.append("---")
            
        lines.append("\nTASK: Audit this safety state. Confirm if the clinical picture supports this decision.")
        return "\n".join(lines)

    def _fail_safe_response(self, reason: str) -> FinalTriageDecision:
        return FinalTriageDecision(
            final_level=TriageLevel.UNMATCHED,
            safety_protocol_used="SYSTEM_FAILURE",
            rationale_summary=reason,
            clinical_coherence_score=0.0,
            coherence_explanation="System failed to extract data",
            requires_human_review=True,
            review_triggers=["System Error"],
            guideline_gap_detected=True,
            conflict_resolution_notes="N/A"
        )