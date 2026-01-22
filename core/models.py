from __future__ import annotations
from typing import Optional, Any, Literal, List
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from enum import Enum
from utils.parsing import _age_str_to_months


class PatientContext(BaseModel):
    age_str: str
    sex: Optional[str] = None
    reason_for_consultation: str
    additional: Optional[str] = None
    # We define the field; exclude=True prevents it from being required in the input dict
    age_months: float = Field(default=0.0, init_var=False)

    class Config:
        extra = "allow"

    @model_validator(mode='after')
    def calculate_age_months(self) -> 'PatientContext':
        # This calls your conversion function and sets the attribute
        self.age_months = _age_str_to_months(self.age_str)
        return self

    @property
    def is_neonate(self) -> bool:
        return self.age_months < 1
    
    @property
    def _patient_summary(self) -> str:
        """Build structured summary for AI assessment."""
        summary_parts = [
            f"Chief Complaint: {self.reason_for_consultation}",
            f"Age: {self.age_str}",
            f"Sex: {self.sex}"
        ]
        
        if self.additional:
            summary_parts.append(f"Additional Information: {self.additional}")
        
        return "\n".join(summary_parts)

class TriageLevel(str, Enum):
    """Standardized urgency destinations"""
    # HOME
    PRIMARY_CARE_APPOINTMENT = "primary_care_appointment"
    PRIMARY_CARE_TODAY = "primary_care_today"
    EMERGENCY_DEPARTMENT = "emergency_department"
    UNMATCHED = "unmatched"

# Safety Checker
class RedFlagCategory(str, Enum):
    """Categories of red flag symptoms requiring immediate escalation."""
    AIRWAY_BREATHING = "airway_breathing"
    CIRCULATION = "circulation"
    CONSCIOUSNESS = "consciousness"
    SEVERE_PAIN = "severe_pain"
    PSYCHOLOGICAL = "psychological"
    TRAUMA = "trauma"
    CARDIAC = "cardiac"
    NEUROLOGICAL = "neurological"
    HEMORRHAGE = "hemorrhage"


class SafetyAssessment(BaseModel):
    """Output model for safety assessment with detected red flags."""
    requires_immediate_escalation: bool = Field(
        description="True if case is life-threatening over the next minutes/hours and requires immediate attention"
    )
    detected_red_flags: List[RedFlagCategory] = Field(
        default_factory=list,
        description="List of red flag categories detected in this case"
    )
    reasoning: str = Field(
        description="Explanation of why red flags were chosen and why immediate escalation is needed"
    )

    def format_for_explanation(self) -> str:
        """Format safety assessment for explanation/debugging."""
        lines = ["## SAFETY ASSESSMENT"]
        lines.append(f"Requires Escalation: {self.requires_immediate_escalation}")
        
        if self.detected_red_flags:
            lines.append(f"Red Flags Detected: {', '.join([f.value for f in self.detected_red_flags])}")
        else:
            lines.append("Red Flags Detected: None")
        
        lines.append(f"Reasoning: {self.reasoning}")
        return "\n".join(lines)

# Routing Decision Model
class SpecialtyType(str, Enum):
    """Available medical specialties for triage."""
    FEVER = "fever" 
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    TRAUMA = "trauma"
    GASTROINTESTINAL = "gastrointestinal"
    GENITOURINARY = "genitourinary"
    CARDIOVASCULAR = "cardiovascular"
    OTORHINOLARYNGOLOGY = "otorhinolaryngology"
    OPHTHALMOLOGY = "ophthalmology"
    DERMATOLOGY = "dermatology"
    PSYCHIATRIC = "psychiatric"
    ALLERGY = "allergy"
    TOXICOLOGY = "toxicology"
    ENDOCRINE = "endocrine"
    OTHER = "other"

class RoutingDecision(BaseModel):
    """Output model for routing decision with reasoning."""
    specialties: List[SpecialtyType] = Field(
        description="List of specialties that should evaluate this case. "
                    "Include all relevant specialties based on symptoms and presentation."
    )
    reasoning: str = Field(
        description="Brief clinical reasoning for why these specialties were selected. "
                    "Explain the key symptoms or findings that triggered each specialty."
    )
    
    def format_for_explanation(self) -> str:
        """Format routing decision for explanation/debugging."""
        lines = ["## ROUTING DECISION"]
        lines.append(f"Specialties Selected: {', '.join([s.value for s in self.specialties])}")
        lines.append(f"Reasoning: {self.reasoning}")
        return "\n".join(lines)


# Specialty triage
class RBTriageResult(BaseModel):
    """Standardized triage output"""
    level: TriageLevel
    guideline_reason: str = Field(
        description="guideline grounded reason that justifies the triage"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    extraction_raw: dict = Field(
        description="Raw extraction output for debugging"
    )

    def format_for_explanation(self) -> str:
        """Format rule-based triage result for explanation/debugging."""
        lines = [f"Level: {self.level.value}"]
        lines.append(f"Guideline Reason: {self.guideline_reason}")
        
        if self.extraction_raw:
            lines.append("Extracted Data:")
            for key, value in self.extraction_raw.items():
                if value is not None:
                    lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines)

class SpecialtyTriageResult(BaseModel):
    """Complete result from specialty evaluation including extraction and triage."""
    specialty: SpecialtyType = Field(
        description="The specialty that performed this evaluation"
    )
    triage_result: RBTriageResult = Field(
        description="The triage decision from applying specialty rules"
    )
    extraction_success: bool = Field(
        description="Whether data extraction was successful"
    )
    extraction_error: Optional[str] = Field(
        default=None,
        description="Error message if extraction failed"
    )

    def format_for_explanation(self) -> str:
        """Format specialty triage result for explanation/debugging."""
        lines = [f"## {self.specialty.value.upper()} SPECIALTY"]
        lines.append(f"Extraction Success: {self.extraction_success}")
        
        if self.extraction_error:
            lines.append(f"Extraction Error: {self.extraction_error}")
        else:
            lines.append(self.triage_result.format_for_explanation())
        
        return "\n".join(lines)


class ExtractionStatus(str, Enum):
    """Status of guideline validation for a specialty result."""
    VALID = "valid"  # Rule reasoning matches actual guidelines
    INCOMPLETE_EXTRACTION = "incomplete_extraction"  # Missing key data for full validation
    EXTRACTION_ERRORS = "extraction_errors" # Specific extraction errors (misclassified findings, wrong values, etc.)


class ValidationAction(str, Enum):
    """Possible validation decisions"""
    APPROVE = "approve"
    ESCALATE = "escalate"
    DEESCALATE = "deescalate"
    CHANGE_REASONING = "change_reasoning"
    ASSIGN = "assign"
    UNGUIDELINE = "unguideline"

class SpecialtyValidation(BaseModel):
    """Validation result for a single specialty triage."""
    extraction_status: ExtractionStatus
    extraction_reasoning: str = Field(
        description="Explanation of why the extraction is correct or incorrect, focusing on the most 'controversial' attributes"
    )

    validation_action: ValidationAction
    validated_level: TriageLevel = Field(
        description="Suggested triage level if escalation or de-escalation is needed"
    )
    validation_reasoning: str = Field(
        description="Explanation of why the rule-based result is valid "
    )

    def format_for_explanation(self) -> str:
        """Format specialty validation for explanation/debugging."""
        lines = [f"## EXTRACTION VALIDATION"]
        lines.append(f"  Extraction Status: {self.extraction_status}")
        lines.append(f"  Extraction Reasoning: {self.extraction_reasoning}")
        lines.append(f"  Validation Action: {self.validation_action}")
        lines.append(f"  Validated Level: {self.validated_level}")
        lines.append(f"  Validation Reasoning: {self.validation_reasoning}")
        return "\n".join(lines)

class FinalTriageDecision(BaseModel):
    """The authoritative output of the Aggregator."""
    final_level: TriageLevel
    safety_protocol_used: str = Field(description="Which deterministic rule forced this decision (e.g. 'Max-Urgency-Override')")
    rationale_summary: str = Field(description="A concise summary of why this level was chosen")
    
    # Reliability Metrics
    clinical_coherence_score: float = Field(description="0.0 to 1.0 score of how well symptoms fit together")
    coherence_explanation: str = Field(description="Explanation of the coherence score")
    
    # Audit Flags
    requires_human_review: bool
    review_triggers: List[str] = Field(default_factory=list)
    guideline_gap_detected: bool
    guideline_gaps: List[str] = Field(default_factory=list)
    
    # Conflict Resolution
    conflict_resolution_notes: str = Field(description="How disagreements between specialties were handled")

    def format_for_explanation(self) -> str:
        """Format final triage decision for explanation/debugging."""
        lines = ["=" * 70]
        lines.append("FINAL TRIAGE DECISION")
        lines.append("=" * 70)
        lines.append(f"\nFinal Level: {self.final_level.value}")
        lines.append(f"Safety Protocol Used: {self.safety_protocol_used}")
        lines.append(f"Primary Reasoning: {self.rationale_summary}")
        
        lines.append(f"\n### Quality Indicators")
        lines.append(f"Clinical Coherence Score: {self.clinical_coherence_score:.2f}")
        lines.append(f"Coherence Notes: {self.coherence_explanation}")
        lines.append(f"Requires Human Review: {self.requires_human_review}")
        
        if self.review_triggers:
            lines.append(f"Review Reasons:")
            for reason in self.review_triggers:
                lines.append(f"  - {reason}")
        
        lines.append(f"Falls Outside Guidelines: {self.guideline_gap_detected}")
        if self.guideline_gap_detected:
            lines.append(f"Guideline Gaps:")
            for gap in self.guideline_gaps:
                lines.append(f"  - {gap}")
        
        if self.conflict_resolution_notes:
            lines.append(f"\n### Conflicts Detected and Resolved:\n{self.conflict_resolution_notes}")
        
        return "\n".join(lines)
    
    @classmethod
    def escalation_case(cls, safety_res: SafetyAssessment) -> FinalTriageDecision:
    
        return cls(
            final_level=TriageLevel.EMERGENCY_DEPARTMENT,
            safety_protocol_used='none',
            rationale_summary=safety_res.reasoning,
            clinical_coherence_score=1.0,
            coherence_explanation="Escalation case automatically set to EMERGENCY DEPARTMENT by safety checker.",
            requires_human_review=True,
            review_triggers=[
                "Immediate escalation flagged by safety checker due to detected red flags."
            ],
            guideline_gap_detected=False,
            guideline_gaps=[],
            conflict_resolution_notes='Not done'
        )

class AudienceType(str, Enum):
    """Target audience for the explanation."""
    MEDICAL_STAFF = "medical_staff"  # Doctors, nurses - detailed clinical language
    PATIENT_FAMILY = "patient_family"  # Patients/families - simplified, empathetic
    ADMINISTRATIVE = "administrative"  # Admin staff - concise, procedural
