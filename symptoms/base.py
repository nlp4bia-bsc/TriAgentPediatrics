from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Any, Generic
from pydantic import BaseModel, Field
import json
from outlines.inputs import Chat

from core.models import PatientContext, RBTriageResult, TriageLevel

T = TypeVar('T', bound='SpecialtyModel')


class SpecialtyModel(BaseModel, ABC):
    """Base class for all specialty-specific models"""
    @classmethod
    def get_model_descriptions(cls) -> str:
        """
        Recursively extracts field names and descriptions from a Pydantic model
        to create a comprehensive system prompt guide, handling nested models.
        """
        schema = cls.model_json_schema()
        properties = schema.get("properties", {})
        
        descriptions = []
        
        for field_name, props in properties.items():
            desc = props.get("description", "No specific description provided.")
            
            # 1. Check for nested Pydantic Model reference
            # This checks if the field is a reference to another definition in the schema
            ref = props.get("$ref")
            
            if ref:
                # Extract the definition name from the reference (e.g., '#/definitions/FeverDuration')
                def_name = ref.split("/")[-1]
                
                # Find the actual definition of the nested model
                nested_props = schema.get("definitions", {}).get(def_name, {}).get("properties", {})
                
                # Start the instruction block for the nested model
                descriptions.append(f"- **{field_name}** (Object: {desc}):")
                
                # 2. Add descriptions for the nested model's fields
                for nested_field_name, nested_props_data in nested_props.items():
                    nested_desc = nested_props_data.get("description", "No specific description provided.")
                    # Indent the nested field descriptions for clarity
                    descriptions.append(f"  - **{nested_field_name}**: {nested_desc}")
                
            else:
                # 3. Handle simple, non-nested fields
                descriptions.append(f"- **{field_name}**: {desc}")

        return "\n".join(descriptions)
    
    
    @classmethod
    def get_base_prompt(cls) -> str:
        return """You are a medical information extraction system for triage purposes.
## Your Task
Extract structured medical information from Spanish/Catalan patient consultation reasons into JSON format.

## Core Instructions
1. **Handle Missing Information**: If a finding is not mentioned, set it to `null` (not `false`)
2. **Be Precise**: Extract only information explicitly stated or clearly implied
3. **Use Consistent Units**: Ensure quantity and unit pairs are logically consistent
4. **Follow Examples**: The examples show the expected extraction patterns

## Output Format
Return ONLY valid JSON matching the provided schema. No additional text or explanation.
"""

    @classmethod
    @abstractmethod
    def get_specialty_prompt(cls) -> str:
        pass

    @classmethod
    def get_system_prompt(cls) -> str:
        return cls.get_base_prompt() + cls.get_specialty_prompt()

class SpecialtyRuler(ABC, Generic[T]): # abstract and generic
    """Base class for specialty-specific triage rules"""
    specialty_name: str  # Declare as class attribute

    def __init__(self, patient_ctx: PatientContext, model: T):
        self.patient_ctx = patient_ctx
        self.model = model
        
    @abstractmethod
    def apply_triage(self) -> RBTriageResult:
        """Apply triage rules and return result"""
        pass


    def format_result(self, u_level: TriageLevel, reason: str) -> RBTriageResult:
        """Helper to format triage result"""
        return RBTriageResult(
            level=u_level,
            guideline_reason=reason,
            extraction_raw=json.loads(self.model.model_dump_json())
        )