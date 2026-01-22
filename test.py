import argparse
import asyncio
import time
from typing import Tuple, Optional, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic_ai.models.outlines import OutlinesModel
import pandas as pd

from agents.safety import SafetyChecker
from agents.router import TriageRouter
from agents.orchestrator import SpecialtyOrchestrator, SpecialtyEvaluator
from agents.arbiter import TriageAggregator
from agents.explainer import TriageExplainer

from core.models import (
    PatientContext,
    SafetyAssessment,
    RoutingDecision,
    FinalTriageDecision,
    AudienceType,
)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def header(title: str):
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)


def section(title: str):
    print(f"\n--- {title} ---")


def timing(label: str, seconds: float):
    print(f"{label:<30}: {seconds:>6.2f}s")

def get_fever_cases() -> pd.DataFrame: 
    df = pd.read_csv("data/lay_es.csv") 
    fever_df = df[df['reason'].apply(
        lambda x: any(keyword in x for keyword in ['fiebre', 'febre', 'º', 'caliente', 'temperatura'])
        )
    ] 
    print(len(fever_df), "cases in fever dataset.") 
    fever_df['gender'].fillna(value='female', inplace=True) # there's a fucking case with no gender 
    return fever_df


# ---------------------------------------------------------------------
# Core System
# ---------------------------------------------------------------------

class MedicalTriageSystem:
    def __init__(self, model: OutlinesModel):
        self.safety = SafetyChecker(model)
        self.router = TriageRouter(model)
        self.evaluator = SpecialtyEvaluator(out_model=model)
        self.orchestrator = SpecialtyOrchestrator(self.evaluator)
        self.aggregator = TriageAggregator(model)

    async def triage(
        self,
        patient_ctx: PatientContext
    ) -> Tuple[
        SafetyAssessment,
        Optional[RoutingDecision],
        Optional[Dict],
        FinalTriageDecision
    ]:

        safety_res = await self.safety.check_immediate_escalation(patient_ctx)

        if safety_res.requires_immediate_escalation:
            return (
                safety_res,
                None,
                None,
                FinalTriageDecision.escalation_case(safety_res),
            )

        routing_res = await self.router.route(patient_ctx)

        specialty_results = await self.orchestrator.evaluate_all(
            patient_ctx,
            routing_res.specialties,
        )

        final_decision = await self.aggregator.aggregate(
            patient_ctx,
            specialty_results,
        )

        return safety_res, routing_res, specialty_results, final_decision


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_model(model_name: str, dtype: str) -> OutlinesModel:
    header("Loading model")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch_dtype,
    )
    t1 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t2 = time.time()

    model = OutlinesModel.from_transformers(hf_model, tokenizer)
    t3 = time.time()

    timing("HF model load", t1 - t0)
    timing("Tokenizer load", t2 - t1)
    timing("Outlines wrapper", t3 - t2)

    return model


# ---------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------

async def run(args: argparse.Namespace):

    model = load_model(args.model, args.dtype)
    system = MedicalTriageSystem(model)
    explainer = TriageExplainer(model)

    patient_ctx = PatientContext(
        reason_for_consultation=args.reason,
        age_str=args.age,
        sex=args.sex,
    )

    header("Running triage")

    t0 = time.time()
    safety, routing, specialty_results, decision = await system.triage(patient_ctx)
    t1 = time.time()

    timing("Triage pipeline", t1 - t0)

    header("Generating explanations")

    explanations = await explainer.explain(
        patient_ctx=patient_ctx,
        safety_assessment=safety,
        final_decision=decision,
        audience=args.audience,
        routing_decision=routing,
        specialty_results=specialty_results,
    )

    for audience, text in explanations.items():
        section(f"Explanation for {audience.value}")
        print(text)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Medical Triage System")

    # Model
    parser.add_argument(
        "--model",
        default="inf_models/qwen3_8B_instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )

    # Patient
    parser.add_argument(
        "--reason",
        default="Lleva dos días con fiebre y tiene una tos tan fuerte que acaba vomitando; apenas quiere comer.",
        help="Reason for consultation (free text)",
    )
    parser.add_argument(
        "--age",
        default="11 years",
        help="Age string (e.g. '36 months', '4 years')",
    )
    parser.add_argument(
        "--sex",
        choices=["male", "female"],
        default="female"
    )

    # Explanation
    parser.add_argument(
        "--audience", # ['MEDICAL_STAFF', 'PATIENT_FAMILY', 'ADMINISTRATIVE']
        nargs="+",
        default=[AudienceType.MEDICAL_STAFF],
        type=lambda x: AudienceType[x],
        help=f"Audience(s): {[a.name for a in AudienceType]}",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    for audience in args.audience:
        print("Selected audience:", audience)
    asyncio.run(run(args))
