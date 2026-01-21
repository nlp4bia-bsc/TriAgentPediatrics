from typing import Any, Optional
import re
from word2number import w2n
import json

# general parsing
def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            return float(w2n.word_to_num(value))
        except ValueError:
            return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    
    true_values = ['true', '1', 't', 'yes', 'y', 'on']
    false_values = ['false', '0', 'f', 'no', 'n', 'off', ]
    
    value_str = str(value).lower().strip()
    
    if value_str in true_values:
        return True
    elif value_str in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to boolean")
    
# NuExtract parsing
def _boolify_strings_rec(json_obj: Optional[dict], template: str) -> Optional[dict]:
    """
    NuExtract doesn't have automatic boolean assignations, so we first let it choose between 'True' and 'False', and then use this function to transform it.
    The boolean values might be nested, and so we recursively call this function."""
    if not json_obj:
        return None
    json_template = json.loads(template)
    output_json = {}
    for k, v in json_obj.items():
        if json_template[k] == ["True", "False"]:
            if not v:
                v = False
            else:
                v = _safe_bool(v)
        elif isinstance(v, dict):
            v = _boolify_strings_rec(v, json.dumps(json_template[k]))
        
        output_json[k] = v
    return output_json

# Age formating to work only with months
def _parse_time_str(time_str: str) -> tuple[Optional[float], str]:
    pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(day|hour|week|month|year)s?\b'
    matches = re.findall(pattern, time_str, re.IGNORECASE) # find all, but probably just finds one
    if len(matches) < 1:
        raise ValueError("Age string format is not understood:", time_str)

    m = matches[0]
    return _safe_float(m[0]), m[1]


def _age_str_to_months(age_str: str) -> float:
    if not age_str:
        raise ValueError("Age is not provided")
    unit2mult = {
        'day': 1/30, 'days': 1/30,
        'week': 1/4, 'weeks': 1/4,
        'month': 1, 'months': 1,
        'year': 12, 'years': 12
    }
    try:
        qt, unit = _parse_time_str(age_str)
        if unit in unit2mult:
            return qt * unit2mult[unit]
    except Exception:
        raise ValueError("Age string format is not understood:", age_str)
    return 0


## FEVER parsing utils
def _duration_2_days(qt: float, unit: str) -> Optional[float]:
    if not qt or not unit:
        raise ValueError("Quantity and/or unit not provided")
    
    unit2mult = {
        'minute': 1/1440, 'minuto': 1/1440,
        'hour': 1/24, 'hora': 1/24,
        'day': 1, 'dia': 1,
        'week': 7, 'semana': 7,
        'month': 30, 'mes': 30,
    }
    if unit not in unit2mult:
        raise ValueError("Time unit couldn't be parsed")

    return qt * unit2mult[unit]


def _temp_2_celcius(degrees: float, unit: str) -> Optional[float]:
    if not degrees or not unit:
        raise ValueError("Degrees and/or unit not provided")
    
    celcius_kw = ['celcius', 'ºc', 'c']
    fahrenheit_kw = ['fahrenheit', 'ºf', 'f']
    if unit in celcius_kw:
        return degrees
    elif unit in fahrenheit_kw:
        return (degrees-32) / 1.8
    
    raise ValueError("Couldn't account for {value} as a temperature unit")


## NEUROLOGICAL parsing utils
def _duration_2_min(qt: float, unit: str) -> Optional[float]:
    if not qt or not unit:
        raise ValueError("Quantity and/or unit not provided")
    
    unit2mult = {
        'minute': 1,
        'hour': 60,
        'day': 1,
        'week': 7,
        'month': 30,
    }
    if unit not in unit2mult:
        raise ValueError("Time unit couldn't be parsed")

    return qt * unit2mult[unit]