from ..base import SpecialtyRuler
from .model import FeverModel
from core.models import RBTriageResult, TriageLevel


class FeverRuler(SpecialtyRuler[FeverModel]):
    """Triage rules for fever cases"""
    specialty_name: str = "fever"
    
    def apply_triage(self) -> RBTriageResult:
        """Apply deterministic fever triage rules"""
        
        # URGENT: sepsis signs
        if self.model.has_sepsis_signs:
            return self.format_result(
                TriageLevel.EMERGENCY_DEPARTMENT,
                "fever with sepsis signs"
            )
        
        # URGENT: neonate
        if self.patient_ctx.is_neonate:
            return self.format_result(
                TriageLevel.EMERGENCY_DEPARTMENT,
                "neonate with fever"
            )
        
        if self.patient_ctx.age_months < 3:
            return self.format_result(
                TriageLevel.EMERGENCY_DEPARTMENT,
                "neonate with fever"
            )
        
        # URGENT: immunosuppressed
        if self.model.immunosuppressed:
            return self.format_result(
                TriageLevel.EMERGENCY_DEPARTMENT,
                "fever with vulnerable condition"
            )
        
        # URGENT: suspected tropical disease
        if self.model.recent_tropical_travel:
            return self.format_result(
                TriageLevel.EMERGENCY_DEPARTMENT,
                "fever with suspected tropical disease"
            )
        
        # URGENT: long duration
        duration = self.model.fever_duration_days
        if duration is not None and duration >= 7:
            return self.format_result(
                TriageLevel.EMERGENCY_DEPARTMENT,
                "fever for 7 days or more"
            )
        
        # Age-based rules
        age_months = self.patient_ctx.age_months
        
        if age_months is not None and age_months < 24:
            if duration is not None and duration > 1:
                return self.format_result(
                    TriageLevel.PRIMARY_CARE_TODAY,
                    "fever in child < 2 yrs for more than one day"
                )
            if duration is not None and duration <= 1:
                return self.format_result(
                    TriageLevel.PRIMARY_CARE_APPOINTMENT,
                    "fever in child < 2 yrs for one day or less"
                )
        
        if age_months is not None and age_months >= 24:
            if duration is not None and duration >= 3:
                return self.format_result(
                    TriageLevel.PRIMARY_CARE_TODAY,
                    "fever in child over 2 yrs for 3 days or more"
                )
            if duration is not None and duration < 3:
                return self.format_result(
                    TriageLevel.PRIMARY_CARE_APPOINTMENT,
                    "Fever for less than 3 days in a child over 24 months old without any alarm signs"
                )
        
        # Fallback
        return self.format_result(
            TriageLevel.UNMATCHED,
            "insufficient information for triage"
        )