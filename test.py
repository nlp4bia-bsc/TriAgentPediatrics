from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic_ai.models.outlines import OutlinesModel
import torch
from agents.safety import SafetyChecker
from agents.router import TriageRouter
from agents.orchestrator import SpecialtyOrchestrator, SpecialtyEvaluator
# from agents.aggregator import TriageAggregator
from agents.arbiter import TriageAggregator
from agents.explainer import TriageExplainer
from core.models import PatientContext, SafetyAssessment, RoutingDecision, SpecialtyType, FinalTriageDecision
import time
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm

# class MedicalTriageSystem:
#     def __init__(self, model):
#         self.safety = SafetyChecker(model)
#         self.router = TriageRouter(model)
#         self.evaluator = SpecialtyEvaluator(out_model=model)
#         self.orchestrator = SpecialtyOrchestrator(self.evaluator)
#         self.aggregator = TriageAggregator(model)
#         # self.explainer = TriageExplainer(model)
    
#     async def triage(
#             self, patient_ctx: PatientContext, verbose: bool=False
#                      ) -> Tuple[SafetyAssessment, Optional[RoutingDecision], Optional[Dict[SpecialtyType, Dict[str, Any]]], FinalTriageDecision]:
#         # Safety override
#         escalation_results = await self.safety.check_immediate_escalation(patient_ctx)
#         if verbose: print(escalation_results.format_for_explanation())
#         if escalation_results.requires_immediate_escalation:
#             triage_result = FinalTriageDecision.escalation_case(escalation_results)
#             return (escalation_results, None, None, triage_result)
#         relevant_specialties = await self.router.route(patient_ctx)
#         if verbose: print(relevant_specialties.format_for_explanation())
#         # Parallel specialty evaluation
#         specialty_results = await self.orchestrator.evaluate_all(
#             patient_ctx, 
#             relevant_specialties.specialties
#         )
#         for specialty, specialty_result in specialty_results.items():
#             if specialty_result['status'] == 'success':
#                 if verbose: print(f"{specialty.value} extraction and triage:")
#                 if verbose: print(specialty_result['triage'].format_for_explanation())
#                 if verbose: print(specialty_result['validation'].format_for_explanation())
#         # Aggregate results
#         triage_result = await self.aggregator.aggregate(patient_ctx, specialty_results)
#         if verbose: print(triage_result.format_for_explanation())
#         return (escalation_results, relevant_specialties, specialty_results, triage_result)

class MedicalTriageSystem:
    def __init__(self, model: OutlinesModel):
        self.safety = SafetyChecker(model)
        self.router = TriageRouter(model)
        self.evaluator = SpecialtyEvaluator(out_model=model)
        self.orchestrator = SpecialtyOrchestrator(self.evaluator)
        self.aggregator = TriageAggregator(model)
        self.explainer = TriageExplainer(model)

    async def triage(
        self, 
        patient_ctx: PatientContext, 
        verbose: bool = True
    ) -> Tuple[SafetyAssessment, Optional[RoutingDecision], Optional[Dict], FinalTriageDecision]:
        
        # 1. Safety Gate
        safety_res = await self.safety.check_immediate_escalation(patient_ctx)
        
        if safety_res.requires_immediate_escalation:
            triage_res = FinalTriageDecision.escalation_case(safety_res)
            
            if verbose:
                # Use the explainer to show the debug dashboard even for safety stops
                await self.explainer.explain_for_debugging(
                    patient_ctx, safety_assessment=safety_res, final_decision=triage_res
                )
            return (safety_res, None, None, triage_res)

        # 2. Routing
        routing_res = await self.router.route(patient_ctx)

        # 3. Parallel Specialty Evaluation
        # Assuming evaluate_all returns SpecialtyTriageResult objects
        specialty_results = await self.orchestrator.evaluate_all(
            patient_ctx, 
            routing_res.specialties
        )

        # 4. Aggregation (Consensus)
        triage_res = await self.aggregator.aggregate(patient_ctx, specialty_results)

        # 5. Interpretability Layer (The "Lifecycle" Explanation)
        if verbose:
            # This triggers the Rich Dashboard automatically
            await self.explainer.explain_for_debugging(
                patient_ctx=patient_ctx,
                final_decision=triage_res,
                safety_assessment=safety_res,
                routing_decision=routing_res,
                specialty_results=specialty_results
            )

        return (safety_res, routing_res, specialty_results, triage_res)

async def main_many():
    """Demonstrate how to use the whole pipeline."""
    print("Loading model...")
    MODEL_NAME = "inf_models/qwen3_8B_instruct"  # This can be a local path
    t1 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto", 
        dtype=torch.bfloat16 
    )
    print("Loading tokenizer...")
    t2 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t3 = time.time()
    model = OutlinesModel.from_transformers(hf_model, tokenizer)
    # 2. Wrap it for Pydantic AI using Outlines
    print("Pipeline Agent instantiation...")
    triage_system = MedicalTriageSystem(model)
    t4 = time.time()

    df = pd.read_csv("data/lay_es.csv")
    fever_df = df[df['reason'].apply(lambda x: any(keyword in x for keyword in ['fiebre', 'febre', 'º', 'caliente', 'temperatura']))]
    print(len(fever_df), "cases in fever dataset.")
    fever_df['gender'].fillna(value='female', inplace=True) # there's a fucking case with no gender
    fev_cases: List[PatientContext] = []
    
    for _, (fev_case) in fever_df.iterrows():
        fev_cases.append(
            PatientContext(
                reason_for_consultation=fev_case.reason,
                age_str=fev_case.age,
                sex=fev_case.gender
            )
        )

    time_deltas = []
    decisions = []
    pred_labels = []
    # debug_explanations = []
    # medical_explanations = []
    last_time = time.time()
    for i, fev_case in tqdm(enumerate(fev_cases)):
        escalation_results, relevant_specialties, specialty_results, triage_result = \
            await triage_system.triage(fev_case)
        pred_labels.append(triage_result.final_level.value)
        decisions.append(triage_result)
        time_deltas.append(last_time - time.time())
        last_time = time.time()

    print()
    print("Time Summary:")
    print(f"Model Load Time: {t2 - t1:.2f} seconds")
    print(f"Tokenizer Load Time: {t3 - t2:.2f} seconds")
    print(f"Agent Init Time: {t4 - t3:.2f} seconds")
    print(f"Processing Time stats:\n", pd.Series(time_deltas).describe())
    print("=" * 70)
    
    fever_df['pred_labels'] = pred_labels
    fever_df['decisions'] = decisions
    # fever_df['debug_explanation'] = debug_explanations
    # fever_df['medical_explanation'] = medical_explanations
    fever_df['processing_time'] = time_deltas
    fever_df.to_csv("results/fev_res.csv", index=False)

async def main_simple():
    """Demonstrate how to use the whole pipeline."""
    print("Loading model...")
    MODEL_NAME = "inf_models/qwen3_8B_instruct"  # This can be a local path
    t1 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto", 
        dtype=torch.bfloat16 
    )
    print("Loading tokenizer...")
    t2 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t3 = time.time()
    model = OutlinesModel.from_transformers(hf_model, tokenizer)
    # 2. Wrap it for Pydantic AI using Outlines
    print("Pipeline Agent instantiation...")
    triage_system = MedicalTriageSystem(model)
    t4 = time.time()
    patient_ctx = PatientContext(
        reason_for_consultation="slight fever over 2 days with patient undergoing cancer treatment.",
        age_str="12 years",
        sex="male"
    )

    escalation_results, relevant_specialties, specialty_results, triage_result = \
        await triage_system.triage(patient_ctx, verbose=True)
    t5 = time.time()

    print()
    print("Time Summary:")
    print(f"Model Load Time: {t2 - t1:.2f} seconds")
    print(f"Tokenizer Load Time: {t3 - t2:.2f} seconds")
    print(f"Agent Init Time: {t4 - t3:.2f} seconds")
    print(f"Processing Time Time: {t5 - t4:.2f} seconds")
    print("=" * 70)

    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main_simple())