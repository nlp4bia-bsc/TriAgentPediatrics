"""
Specialty Evaluator for Medical Triage
Orchestrates data extraction and rule-based triage for each specialty.
"""

from typing import TypeVar, Tuple, Type, Dict, Any, Optional, List
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.outlines import OutlinesModel
import asyncio

from core.models import (
    PatientContext, 
    RBTriageResult, 
    TriageLevel, 
    SpecialtyType, 
    SpecialtyTriageResult,
    SpecialtyValidation
)

# Import your specialty implementations
from symptoms.base import SpecialtyModel
from symptoms.fever.model import FeverModel
from symptoms.fever.ruler import FeverRuler

T = TypeVar('T', bound=SpecialtyModel)

SPECIALTY_CONFIGS: Dict[SpecialtyType, Dict[str, Type]] = {
    SpecialtyType.FEVER: {
        "model": FeverModel,
        "ruler": FeverRuler
    },
    # Add more specialties as implemented:
    # SpecialtyType.RESPIRATORY: {"model": RespiratoryModel, "ruler": RespiratoryRuler},
}

def is_specialty_implemented(specialty: SpecialtyType) -> bool:
    """Check if a specialty has been implemented."""
    return specialty in SPECIALTY_CONFIGS

VALIDATION_SYSTEM_PROMPT = """You are a Senior Triage Auditor. Your role is to assess whether rule-based triage decisions align with actual clinical guidelines. These decisions are based on structured extractions from patient cases and deterministic triage logic applied to said extractions.

## Your Tasks

### 1. Extraction Validation
Assess whether the rule-based system correctly extracted clinical information:
- Are the extracted values accurate representations of the patient's condition?
- Is any critical information missing that should have been captured (incomplete)?
- Are there extraction errors (misclassifications, wrong values)?\n

** Decision Categories: **
- VALID: Extraction is accurate and complete
- INCOMPLETE_EXTRACTION: Some key data is missing (attribute not set to a value, not missing attribute) for full validation
- EXTRACTION_ERRORS: Specific extraction errors (misclassified findings, wrong values, etc.)

### 2. Reasoning Validation
Assess whether the triage reasoning is correct for THIS SPECIFIC symptom group:
- Does the reasoning correctly apply the relevant guideline rules for this case?
- If the reasoning is flawed, identify the guideline bulletpoint that was misapplied or overlooked.
- Suggest the correct triage level if the original is inappropriate.

** Decision Categories: **
- APPROVE: Reasoning is correct
- ESCALATE: There is a guideline rule that better fits this case which is in an upper urgency group
- DEESCALATE: There is a guideline rule that better fits this case which is in an lower urgency group
- CHANGE_REASONING: There is a guideline rule that better fits this case which is the same urgency group
- ASSIGN: Rule based triage outputs UNMATCHED and case fits within a guideline bulletpoint
- UNGUIDELINE: Case presents a condition for this specific symptom group that is not considered in the guidelines

## Output Requirements
For each specialty validation, provide:
- `extraction_status`: One of VALID / INCOMPLETE_EXTRACTION / EXTRACTION_ERRORS
- `validation_action`: One of APPROVE / ESCALATE / DEESCALATE / CHANGE_REASONING / ASSIGN / UNGUIDELINE
- `validated_level`: If you disagree with the original level, suggest the correct one
- `validation_reasoning`: Clear explanation of your assessment

"IMPORTANT: When filling out reasoning fields, BE CONCISE. "
"Limit specific reasoning fields to 1-2 sentences. "
"Do not repeat information already present in other fields."
"""

class SpecialtyEvaluator:
    """
    Evaluates a patient case for a specific specialty using:
    1. Pydantic AI Agent for data extraction (LLM + constrained generation)
    2. Deterministic rule-based triage logic
    """
    
    def __init__(self, out_model: OutlinesModel):
        """
        Initialize evaluator with HuggingFace model.
        
        Args:
            out_model: OutlinesModel instance for constrained generation
        """
        self.out_model = out_model
        # Pre-create agents for each registered specialty (for efficiency)
        self.extraction_agents: Dict[SpecialtyType, Agent[None, SpecialtyModel]] = {}
        self.validation_agents: Dict[SpecialtyType, Agent[None, SpecialtyValidation]] = {}

    def initialize_agents(self, specialties: List[SpecialtyType]):
        """Pre-create agents for all registered specialties."""
        for specialty in specialties:
            if is_specialty_implemented(specialty):
                model_class = SPECIALTY_CONFIGS[specialty]["model"]
                
                self.extraction_agents[specialty] = Agent(
                    model=self.out_model,
                    output_type=model_class,
                    system_prompt=SpecialtyModel.get_base_prompt(),
                    retries=2,
                )

            self.validation_agents[specialty] = Agent(
                model=self.out_model,
                output_type=SpecialtyValidation,
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                retries=2,
            )

    async def run_specialty_triage(
        self, 
        patient_ctx: PatientContext, 
        specialty: SpecialtyType
    ) -> Tuple[SpecialtyTriageResult, Optional[SpecialtyValidation]]:
        """
        Factor 7: Self-Contained Execution.
        This wraps the sequential logic for ONE specialty.
        """
        # 1. Evaluate (Extraction + Rules)
        triage_result = await self.evaluate(patient_ctx, specialty)
        validation_result = await self.validate_extraction(
            patient_ctx, 
            specialty, 
            triage_result
        )
        return triage_result, validation_result
    
    async def evaluate(
        self,
        patient_ctx: PatientContext,
        specialty: SpecialtyType
    ) -> SpecialtyTriageResult:
        """
        Evaluate a patient case for a specific specialty.
        
        Args:
            patient_ctx: Patient context with reason for consultation
            specialty: Which specialty to evaluate
        
        Returns:
            SpecialtyTriageResult with extraction and triage outcome
        """
        # Validate specialty is registered
        if specialty not in SPECIALTY_CONFIGS:
            return self._unmatched_result(
                specialty, 
                f"Specialty {specialty.value} not implemented"
            )
        
        config = SPECIALTY_CONFIGS[specialty]
        model_class = config["model"]
        ruler_class = config["ruler"]
        
        # Step 1: Extract structured data using Agent
        try:
            extracted_model = await self._extract_data(patient_ctx, model_class, specialty)
        except Exception as e:
            return self._error_result(specialty, str(e))
        
        # Step 2: Apply deterministic triage rules
        ruler = ruler_class(patient_ctx=patient_ctx, model=extracted_model)
        triage_result = ruler.apply_triage()
        
        return SpecialtyTriageResult(
            specialty=specialty,
            triage_result=triage_result,
            extraction_success=True,
            extraction_error=None
        )
    
    async def _extract_data(
        self,
        patient_ctx: PatientContext,
        model_class: Type[T],
        specialty: SpecialtyType
    ) -> SpecialtyModel:
        """
        Extract structured data using pre-created Pydantic AI Agent.
        """
        # Get the pre-created agent for this specialty
        agent = self.extraction_agents[specialty]
        
        # Build prompt: specialty instructions + patient summary
        prompt = self._build_extraction_prompt(patient_ctx, model_class)
        
        # Run agent with constrained generation
        result = await agent.run(
            prompt,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 300})
        )
        
        return result.output
    
    def _build_extraction_prompt(
        self, 
        patient_ctx: PatientContext, 
        model_class: Type[T]
    ) -> str:
        """Build the complete extraction prompt."""
        specialty_instructions = model_class.get_specialty_prompt()
        patient_summary = patient_ctx._patient_summary
        
        return f"{specialty_instructions}\n\nPatient Case:\n{patient_summary}"
    
    def _unmatched_result(self, specialty: SpecialtyType, reason: str) -> SpecialtyTriageResult:
        """Helper for unmatched specialty results."""
        return SpecialtyTriageResult(
            specialty=specialty,
            triage_result=RBTriageResult(
                level=TriageLevel.UNMATCHED,
                guideline_reason=reason,
                extraction_raw={}
            ),
            extraction_success=False, # set this for true in order to be able to process in the aggregator
            extraction_error=reason
        )
    
    def _error_result(self, specialty: SpecialtyType, error: str) -> SpecialtyTriageResult:
        """Helper for extraction error results."""
        return SpecialtyTriageResult(
            specialty=specialty,
            triage_result=RBTriageResult(
                level=TriageLevel.UNMATCHED,
                guideline_reason=f"Extraction failed: {error}",
                extraction_raw={}
            ),
            extraction_success=False,
            extraction_error=error
        )
    
    async def validate_extraction(
        self,
        patient_ctx: PatientContext,
        specialty: SpecialtyType,
        triage_result: SpecialtyTriageResult,
    ) -> SpecialtyValidation:
        """
        Validate the extraction quality for a specialty triage result.
        
        Args:
            patient_ctx: Patient context
            specialty: Specialty type
            triage_result: Rule-based triage result
        Returns:
            SpecialtyValidation with validation details
        """
        prompt = self._build_validation_prompt(
            patient_ctx,
            specialty,
            triage_result
        )
        result = await self.validation_agents[specialty].run(
            prompt,
            model_settings=ModelSettings(extra_body={'max_new_tokens': 400})
        )
        return result.output
    

    def _build_validation_prompt(
        self,
        patient_ctx: PatientContext,
        specialty: SpecialtyType,
        rb_triage_result: SpecialtyTriageResult
    ) -> str:
        """Build prompt for validating a specialty result."""
        return (f"""Validate this triage result:

<patient_case>\n{patient_ctx._patient_summary}</patient_case>

<specialty>{specialty.value}</specialty>
<rule_based_triage>{rb_triage_result.triage_result.level.value}</rule_based_triage>
<rule_based_reason>{rb_triage_result.triage_result.guideline_reason}</rule_based_reason>
""" + (f"""
<extracted_fields>\n{SPECIALTY_CONFIGS[specialty]['model'].get_model_descriptions() if is_specialty_implemented(specialty) else None}</extracted_fields>
<extracted_data>\n{self._format_extraction(rb_triage_result.triage_result.extraction_raw)}</extracted_data>
"""
        if is_specialty_implemented(specialty)
        else """
The case is unmatched because extraction and rule construction is not yet built for this specialty, so create triage based only on the guidelines.
In this case, the first two output sections will always be:
extraction_status: INCOMPLETE_EXTRACTION
validation_action: ASSIGN or UNGUIDELINE

Try to find the findings of the case that belong to analysed symptomatic group and match it to an appropiate bulletpoint in the guidelines, if any.
- If macthed, use that match to determine the triage level
- Otherwise, try to follow the same urgency rationale in the guidelines to inform the triage classification.
""") + 
f"""
<{specialty.value.lower()}_guidelines>\n{open(f'guidelines/{specialty.value.lower()}.md', 'r').read()}</{specialty.value.lower()}_guidelines>

Provide your validation assessment.
"""
)
    
    def _format_extraction(self, extraction_raw: dict) -> str:
        """Format extracted data for prompt."""
        lines = []
        for key, value in extraction_raw.items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines) if lines else "  (no data extracted)"


class SpecialtyOrchestrator:
    def __init__(self, evaluator: SpecialtyEvaluator):
        self.evaluator = evaluator

    async def evaluate_all(
        self,
        patient_ctx: PatientContext,
        specialties: List[SpecialtyType]
    ) -> Dict[SpecialtyType, Dict[str, Any]]:
        """
        Executes all specialties concurrently. 
        Each specialty's internal steps are sequential.
        """
        self.evaluator.initialize_agents(specialties)
        
        # Factor 11: Comprehensive Logging/Telemetry
        # We use TaskGroup (Python 3.11+) for robust concurrency
        results = {}
        
        async with asyncio.TaskGroup() as tg:
            # Create a dedicated task for each specialty
            tasks = {
                spec: tg.create_task(
                    self.evaluator.run_specialty_triage(patient_ctx, spec)
                )
                for spec in specialties
            }

        # Once the TaskGroup finishes, all tasks are done or one has failed
        for spec, task in tasks.items():
            try:
                triage, validation = task.result()
                results[spec] = {
                    "triage": triage,
                    "validation": validation,
                    "status": "success"
                }
            except Exception as e:
                results[spec] = {
                    "status": "error",
                    "error": str(e)
                }
                
        return results