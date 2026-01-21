from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic_ai.models.outlines import OutlinesModel
import torch
from agents.safety import SafetyChecker
from agents.router import TriageRouter
from agents.orchestrator import SpecialtyOrchestrator, SpecialtyEvaluator
from agents.aggregator import TriageAggregator
from agents.explainer import TriageExplainer
from core.models import PatientContext,TriageLevel, AudienceType, FinalTriageDecision , TriageExplanation
import time
import pandas as pd
from typing import Tuple, List

class MedicalTriageSystem:
    def __init__(self, model):
        self.safety = SafetyChecker(model)
        self.router = TriageRouter(model)
        self.evaluator = SpecialtyEvaluator(out_model=model)
        self.orchestrator = SpecialtyOrchestrator(self.evaluator)
        self.aggregator = TriageAggregator(model)
        # self.explainer = TriageExplainer(model)
    
    async def triage(self, patient_ctx: PatientContext) -> Tuple[FinalTriageDecision, str, str]:# -> Tuple[FinalTriageDecision, str, TriageExplanation]:
        # Safety override
        escalation_results = await self.safety.check_immediate_escalation(patient_ctx)
        if escalation_results.requires_immediate_escalation:
            triage_result = FinalTriageDecision.escalation_case(escalation_results)
            # debugging_explanation = await self.explainer.explain_for_debugging(
            #     patient_ctx,
            #     triage_result,
            #     safety_assessment=escalation_results
            # )

            # medical_explanation = await self.explainer.explain(
            #     patient_ctx,
            #     triage_result,
            #     safety_assessment=escalation_results,
            #     audience=AudienceType.MEDICAL_STAFF
            # )
            return (triage_result, "debugging_explanation", "medical_explanation")
        # print("Safety check passed.")
        # print("Passed Reason:", escalation_results.reasoning)
        
        # print("Routing patient case...")
        # Route to specialties
        relevant_specialties = await self.router.route(patient_ctx)
        
        # Parallel specialty evaluation
        specialty_results = await self.orchestrator.evaluate_all(
            patient_ctx, 
            relevant_specialties.specialties
        )
        
        # Aggregate results
        final_decision = await self.aggregator.aggregate(patient_ctx, specialty_results)
        print(final_decision.format_for_explanation())
        # debugging_explanation = await self.explainer.explain_for_debugging(
        #     patient_ctx,
        #     final_decision=final_decision,
        #     validated_specialties=specialty_results
        # )
        
        # medical_explanation = await self.explainer.explain(
        #     patient_ctx,
        #     final_decision=final_decision,
        #     validated_specialties=specialty_results,
        #     audience=AudienceType.MEDICAL_STAFF
        # )

        return (final_decision, 'debugging_explanation', 'medical_explanation')

async def example_usage():
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
    that_one_sample = fever_df[fever_df['reason']=='Tiene mucho dolor al orinar y orina con sangre, aunque no tiene fiebre.']
    fever_df = fever_df.sample(5, random_state=32)
    fever_df = pd.concat([fever_df, that_one_sample])
    print(len(fever_df), "cases in fever dataset.")

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
    # debug_explanations = []
    # medical_explanations = []
    last_time = time.time()
    for i, fev_case in enumerate(fev_cases):
        print(f"CASE {i}:")
        print(fev_case._patient_summary)
        print("=" * 70)
        decision, debug_explanation, medical_explanation = await triage_system.triage(fev_case)
        decisions.append(decision)
        # debug_explanations.append(debug_explanation)
        # medical_explanations.append(medical_explanation)
        time_deltas.append(last_time - time.time())
        last_time = time.time()
        # print("MEDICAL STAFF EXPLANATION")
        # print("=" * 70)
        
        # print(f"\nSummary: {medical_explanation.summary}")
        # print(f"\nDetailed Explanation:\n{medical_explanation.detailed_explanation}")
        # print(f"\nKey Findings:")
        # for finding in medical_explanation.key_findings:
        #     print(f"  - {finding}")
        # print(f"\nNext Steps: {medical_explanation.next_steps}")
        
        
        # print("\n" + "=" * 70)
        # print("DEBUG OUTPUT")
        # print("=" * 70)
        
        # print(debug_explanation)
    print()
    print("Time Summary:")
    print(f"Model Load Time: {t2 - t1:.2f} seconds")
    print(f"Tokenizer Load Time: {t3 - t2:.2f} seconds")
    print(f"Agent Init Time: {t4 - t3:.2f} seconds")
    print(f"Processing Time stats:\n", pd.Series(time_deltas).describe())
    print("=" * 70)
    
    fever_df['decisions'] = decisions
    # fever_df['debug_explanation'] = debug_explanations
    # fever_df['medical_explanation'] = medical_explanations
    fever_df['processing_time'] = time_deltas
    fever_df.to_csv("results/fev_res.csv", index=False)
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())