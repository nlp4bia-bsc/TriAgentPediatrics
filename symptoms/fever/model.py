from ..base import SpecialtyModel
from utils.parsing import _duration_2_days, _temp_2_celcius
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

EXAMPLE_CASES = [
    "Hace tres días se le hinchó el párpado y no se le ha deshinchado, y estas últimas 12 horas ha empezado con fiebre (38,3ºC). También le ha aparecido unas manchas.",
    "Desde hace 12 horas está muy decaído y con mal color, aunque no tiene fiebre.",
    "Lleva semana y media con fiebre y tiene una tos tan fuerte que acaba vomitando; apenas quiere comer y tiene diabetes"
]

EXAMPLE_RESPONSES = [
    '{"fever_duration": {"quantity": 12.0, "time_unit": "hora"}, "fever_temperature": {"degrees": 38.3, "temp_unit": "celcius"}, "lethargy": None, "irritability": None, "non_blanching_rash": None, "cold_extremities": None, "pale_skin": None, "weak_pulses": None, "immunosuppressed": None, "recent_tropical_travel": None}',
    '{"fever_duration": None, "fever_temperature": None, "lethargy": True, "irritability": None, "non_blanching_rash": None, "cold_extremities": None, "pale_skin": True, "weak_pulses": None, "immunosuppressed": None, "recent_tropical_travel": None}',
    '{"fever_duration": {"quantity": 1.5, "time_unit": "semana"}, "fever_temperature": {"degrees": 38.0, "temp_unit": "celcius"}, "lethargy": None, "irritability": None, "non_blanching_rash": None, "cold_extremities": None, "pale_skin": None, "weak_pulses": None, "immunosuppressed": True, "recent_tropical_travel": None}'
]


class FeverDuration(BaseModel):
    quantity: float = Field(
        ge=0, 
        description="The numeric value representing how long the fever has lasted."
    )
    time_unit: Literal["hora", "dia", "semana", "mes"] = Field(
        description="The unit of time associated with the duration quantity."
    )

class Temperature(BaseModel):
    degrees: float = Field(
        ge=36, # 86 ºF
        le=115, # 46 ºC
        description="The recorded body temperature value."
    )
    temp_unit: Literal["celcius", "fahrenheit"] = Field(
        description="The scale used for the temperature measurement."
    )

class FeverModel(SpecialtyModel):
    # # Nested Models
    fever_duration: Optional[FeverDuration] = Field(
        default=None, 
        description="The total duration of the fever episode."
    )
    fever_temperature: Optional[Temperature] = Field(
        default=None, 
        description="The specific temperature reading if available."
    )

    lethargy: Optional[bool] = Field(
        default=None, 
        description="True if patient exhibits extreme tiredness or lack of energy."
    )
    irritability: Optional[bool] = Field(
        default=None, 
        description="True if patient (especially a child) is easily annoyed or angered."
    )
    non_blanching_rash: Optional[bool] = Field(
        default=None, 
        description="True if a rash is present that does not fade under pressure (glass test). Distinguish from generic rashes if no additional information is provided on the observation on the coloration difference after applying pressure or if it is a non-blanching rash by nature (Petechiae)."
    )
    cold_extremities: Optional[bool] = Field(
        default=None, 
        description="True if patient reports abnormal cold in hands and feet despite the presence of fever."
    )
    pale_skin: Optional[bool] = Field(
        default=None, 
        description="True if skin appears unusually light or washed out (pallor)."
    )
    weak_pulses: Optional[bool] = Field(
        default=None, 
        description="True if peripheral pulses are difficult to palpate or thready."
    )
    immunosuppressed: Optional[bool] = Field(
        default=None, 
        description="True if patient has a compromised immune system (e.g., chemotherapy, steroids, HIV)."
    )
    recent_tropical_travel: Optional[bool] = Field(
        default=None, 
        description="True if patient has traveled to tropical regions recently."
    )

    model_config = ConfigDict(extra="forbid")
    
    @property
    def has_poor_perfusion(self) -> bool:
        return any([self.cold_extremities, self.pale_skin, self.weak_pulses])

    @property
    def has_sepsis_signs(self) -> bool:
        return any([
            self.lethargy, self.irritability, 
            self.non_blanching_rash, self.has_poor_perfusion
        ])
    
    @property
    def fever_duration_days(self) -> Optional[float]:   
        if self.fever_duration is None:
            return None
        return _duration_2_days(
            self.fever_duration.quantity, 
            self.fever_duration.time_unit
        )

    @property
    def temperature_celcius(self) -> Optional[float]:
        if self.fever_temperature is None:
            return None
        return _temp_2_celcius(
            self.fever_temperature.degrees,
            self.fever_temperature.temp_unit
        )
    
    @classmethod
    def get_specialty_prompt(cls) -> str:
        user_prompt = (
                "If a finding is not present, set it to None, not False\n"
                "If fever is mentioned as present but not the temperature or duration, set to 38 degrees celcius and 1 day respectively by default.\n"
                "If it is vaguely mentioned, provide a sensible value ('about a week' can be understood as a week or 'mild fever' can be understood to be 37 ºC).\n"
                "If the case **doesn't mention fever or explicitly mentions it isn't present** (e.g.: 'no tiene fiebre', 'sin fiebre'), leave duration and temperature findings as None.\n"
                "You must ensure the time and temperature units match the quantity (e.g. if case defines 2 weeks, it is either qt: 2, unit:'week' or qt:14, unit: 'day').\n"
            )

        # Add the examples section
        for i, (example_case, example_response) in enumerate(zip(EXAMPLE_CASES, EXAMPLE_RESPONSES)):
            user_prompt += (
                f"Example {i}:\n"
                f"Case: {example_case}\n"
                f"Response: {example_response}\n"
            )
        return user_prompt