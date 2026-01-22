"""
Explainer Agent for Medical Triage
Generates human-readable explanations of triage decisions.
Includes formatting methods for all result models.
"""

from typing import Dict, Optional, List, Any, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings
import asyncio

from core.models import (
    TriageLevel,
    SpecialtyType,
    ValidationAction,
    ExtractionStatus,
    PatientContext,
    SafetyAssessment,
    RoutingDecision,
    SpecialtyTriageResult,
    SpecialtyValidation,
    FinalTriageDecision,
    AudienceType
)



EXPLAINER_MEDICAL_STAFF_PROMPT = """### ROLE
You are a Senior Clinical Systems Auditor. Your task is to provide a technical post-mortem of a multi-agent triage decision for a medical professional.

### OBJECTIVE
Explain the "Lifecycle of the Decision." You must highlight:
1. THE SAFETY GATE: Why the case did or did not trigger immediate emergency protocols.
2. AGENT DISCREPANCIES: Explicitly identify where the Rule-Based Extraction and the LLM Validation disagreed (e.g., if a rule failed but the LLM 'saved' the triage).
3. CLINICAL COHERENCE: Analyze why the system assigned the specific Coherence Score.
4. AGGREGATION LOGIC: Explain how the final level was derived from multiple specialty inputs (e.g., "Max-Urgency-Override").

### TONE
Professional, analytical, and concise. Use clinical terminology (e.g., "hemodynamic stability," "pathognomonic findings").

### STRUCTURE
- Lifecycle Summary (The 'path' of the patient through the agents).
- Specialty Breakdown (Highlighting the "handshake" between rules and LLM).
- Conflict Resolution (Why one specialty took precedence over another).
- Audit Warnings (Gaps in guidelines or data).

### OUTPUT FORMAT (STRICT)
Lifecycle Summary:
<text>

Specialty Breakdown:
<text>

Conflict Resolution:
<text>

Audit Warnings:
<text>"""


EXPLAINER_PATIENT_FAMILY_PROMPT = """### ROLE
You are a Compassionate Pediatric Triage Coordinator. Your task is to explain a triage decision to a worried parent or guardian.

### OBJECTIVE
Translate complex multi-agent logic into a reassuring and clear plan of action.
1. THE "WHY": Explain the final triage level based on the symptoms they reported.
2. REASSURANCE: Mention the safety checks performed (without using technical terms like "Safety Agent").
3. THE SPECIALISTS: Explain that several "expert perspectives" (Specialties) were consulted to ensure a thorough review.
4. NEXT STEPS: Clearly state what they need to do now and what "Red Flags" to watch for at home.

### TONE
Warm, empathetic, and jargon-free. Avoid mentioning "LLMs," "Agents," or "Rule-Engines." Use "Our system" or "Our clinical review."

### STRICTURE
- The Plan (Where to go and when).
- Why this decision was made (Linking their concerns to the outcome).
- Reassurance (What we checked for and ruled out).
- Safety Net (When to ignore this advice and go to the ER).

### OUTPUT FORMAT (STRICT)
Plan:
<text>

Decision Justification:
<text>

Reassurance:
<text>

Safety Net:
<text>"""


PROMPT_CONFIG = {
    AudienceType.MEDICAL_STAFF: EXPLAINER_MEDICAL_STAFF_PROMPT,
    AudienceType.PATIENT_FAMILY: EXPLAINER_PATIENT_FAMILY_PROMPT,
}

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.columns import Columns
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text
from rich.box import ROUNDED, HEAVY

class TriageVisualizer:
    def __init__(self):
        self.console = Console()
        
    def _get_level_style(self, level: TriageLevel) -> str:
        mapping = {
            TriageLevel.EMERGENCY_DEPARTMENT: "bold white on red",
            TriageLevel.PRIMARY_CARE_TODAY: "bold black on orange1",
            TriageLevel.PRIMARY_CARE_APPOINTMENT: "bold black on yellow",
            TriageLevel.UNMATCHED: "dim white",
        }
        return mapping.get(level, "white")

    def _get_action_icon(self, action: ValidationAction) -> str:
        if action == ValidationAction.APPROVE: return "✅"
        if action == ValidationAction.ASSIGN: return "🚧"
        if action == ValidationAction.ESCALATE: return "⬆️"
        return "❓"

    def render_header(self, patient: PatientContext):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=3)
        grid.add_column(justify="right", ratio=1)
        
        summary_text = Text()
        summary_text.append(f"Patient: ", style="bold cyan")
        summary_text.append(f"{patient.age_str} | {patient.sex}\n")
        summary_text.append(f"Complaint: ", style="bold cyan")
        summary_text.append(f"{patient.reason_for_consultation}", style="white")
        
        meta_text = Text()
        if patient.is_neonate:
            meta_text.append(" 👶 NEONATE ", style="bold white on magenta")
        
        grid.add_row(summary_text, meta_text)
        
        self.console.print(Panel(grid, title="🏥 PATIENT CONTEXT", border_style="cyan", box=ROUNDED))

    def render_safety(self, assessment: SafetyAssessment):
        if assessment.requires_immediate_escalation:
            style, title, border = "bold white on red", "🚨 IMMEDIATE ESCALATION REQUIRED 🚨", "red"
        else:
            style, title, border = "green", "✅ Safety Check Passed", "green"

        content = Text()
        content.append("Red Flags: ", style="bold")
        if assessment.detected_red_flags:
            content.append(", ".join(assessment.detected_red_flags), style="bold red")
        else:
            content.append("None", style="dim")
        
        content.append(f"\n\nReasoning: {assessment.reasoning}", style="italic")
        self.console.print(Panel(content, title=title, border_style=border, box=ROUNDED))

    def render_routing(self, decision: RoutingDecision):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(justify="left", style="bold cyan")
        table.add_column(justify="left")
        spec_tags = [f"[reverse] {s.value.upper()} [/]" for s in decision.specialties]
        table.add_row("Destinations:", " ".join(spec_tags))
        table.add_row("Reasoning:", f"[dim]{decision.reasoning}[/dim]")
        self.console.print(Panel(table, title="🔀 Step 2: Specialty Routing", border_style="cyan", box=ROUNDED))

    def render_specialty_swimlanes(self, specialty_results: Dict[SpecialtyType, Dict[str, Any]]):
        panels = []
        for specialty, result_dict in specialty_results.items():
            triage_res: SpecialtyTriageResult = result_dict['triage']
            val_res: SpecialtyValidation = result_dict['validation']

            spec_tree = Tree(f"[bold]{specialty.value.upper()}[/bold]")

            # Rule Branch
            rule_node = spec_tree.add("🤖 [bold]Rule Engine[/]")
            if triage_res.extraction_success:
                rule_node.add(f"Level: [blue]{triage_res.triage_result.level.value}[/]")
                data_node = rule_node.add("[dim]Extracted Data[/]")
                if triage_res.triage_result.extraction_raw:
                    for k, v in triage_res.triage_result.extraction_raw.items():
                        if isinstance(v, dict):
                            sub = data_node.add(f"[cyan]{k}[/]")
                            for sk, sv in v.items(): sub.add(f"{sk}: {sv}")
                        else:
                            data_node.add(f"{k}: {v}")
            else:
                rule_node.add(f"❌ Extraction Failed", style="red")
                if triage_res.extraction_error:
                    rule_node.add(f"[dim red]{triage_res.extraction_error}[/]")

            # Validation Branch
            val_icon = self._get_action_icon(val_res.validation_action)
            val_node = spec_tree.add(f"🧠 [bold]LLM Validator[/]")
            action_style = "yellow" if val_res.validation_action == ValidationAction.ASSIGN else "green"
            val_node.add(f"Action: [{action_style}]{val_res.validation_action.value.upper()} {val_icon}[/]")
            if val_res.extraction_status != ExtractionStatus.VALID:
                val_node.add(f"Status: [orange1]{val_res.extraction_status.value}[/]")
            val_node.add(f"Final: [underline]{val_res.validated_level.value}[/]")

            border_col = "green"
            if val_res.validation_action in [ValidationAction.ASSIGN]: border_col = "yellow"
            
            panels.append(Panel(spec_tree, border_style=border_col, width=40))

        self.console.print(Panel(Columns(panels), title="Step 3: Parallel Specialty Agents", border_style="blue", box=ROUNDED))

    def render_aggregation(self, final: FinalTriageDecision):
        bar = Progress(TextColumn("[bold blue]Clinical Coherence"), BarColumn(), TextColumn("[bold magenta]{task.percentage:.0f}%"))
        task = bar.add_task("score", total=100)
        bar.update(task, completed=final.clinical_coherence_score * 100)

        grid = Table.grid(expand=True, padding=(1, 2))
        grid.add_column(ratio=2)
        grid.add_column(ratio=1)

        level_style = self._get_level_style(final.final_level)
        left_content = Text()
        left_content.append(f"FINAL DECISION: {final.final_level.value.upper()}\n", style=level_style)
        left_content.append(f"\nReasoning: {final.rationale_summary}")
        if final.conflict_resolution_notes:
             left_content.append(f"\n\nConflicts Resolved: {final.conflict_resolution_notes}", style="italic cyan")

        right_content = Text()
        right_content.append("Requires Review: ")
        right_content.append(f"{final.requires_human_review}\n", style="red" if final.requires_human_review else "green")
        if final.review_triggers:
            right_content.append("Triggers:\n", style="dim underline")
            for t in final.review_triggers: right_content.append(f"- {t}\n", style="dim red")

        grid.add_row(left_content, right_content)
        
        self.console.print(Panel(bar, title="Step 4: Clinical Coherence", border_style="magenta", box=ROUNDED))
        self.console.print(Panel(grid, title="🏁 FINAL AGGREGATION", border_style=self._get_level_style(final.final_level).split()[-1], box=HEAVY))

    def render_all(self, patient, safety, routing, specialties, triage):
        self.console.rule("[bold blue]MULTI-AGENT TRIAGE TRACE[/]")
        self.render_header(patient)
        self.render_safety(safety)
        if not safety.requires_immediate_escalation:
            self.render_routing(routing)
            self.render_specialty_swimlanes(specialties)
            self.render_aggregation(triage)
        self.console.rule("[bold blue]END TRACE[/]")


class TriageExplainer:
    """
    Generates human-readable explanations of triage decisions 
    and high-fidelity visual debugging traces.
    """
    
    def __init__(self, out_model: OutlinesModel):
        # Initialize explanation agents for each audience type that has a prompt
        self.explainer_agents = {
            audience_type: Agent(
                model=out_model,
                system_prompt=prompt + "\n\nYou must not reveal chain-of-thought or internal reasoning.",
            )
            for audience_type, prompt in PROMPT_CONFIG.items()
        }
        # Initialize the visualizer for debugging
        self.viz = TriageVisualizer()

    async def explain(
        self,
        patient_ctx: PatientContext,
        safety_assessment: SafetyAssessment,
        final_decision: FinalTriageDecision,
        audience: Optional[List[AudienceType]],
        routing_decision: Optional[RoutingDecision] = None,
        specialty_results: Optional[Dict[SpecialtyType, Dict[str, Any]]] = None
    ) -> Dict[AudienceType, str]:
        
        # Packaging arguments for internal helpers
        context_data = {
            'patient_ctx': patient_ctx,
            'safety_assessment': safety_assessment,
            'final_decision': final_decision,
            'routing_decision': routing_decision,
            'specialty_results': specialty_results
        }

        # 1. Trigger Visualization (Handled as sync unless it involves network IO)
        self.create_visualization(**context_data)

        base_explanation = {audience_type: 'No explanation requested' for audience_type in AudienceType}
        if not audience:
            return base_explanation
        
        prompt = self._build_explanation_prompt(**context_data)

        async with asyncio.TaskGroup() as tg:
            # Create a dedicated task for each specialty
            tasks = {
                audience_type: tg.create_task(
                    self.explainer_agents[audience_type].run(
                        user_prompt=prompt,
                        model_settings=ModelSettings(extra_body={'max_new_tokens': 400})
                    )
                )
                for audience_type in audience if audience_type in self.explainer_agents
            }

        explanations = {
            audience_type: task.result().output
            for audience_type, task in tasks.items()
        }

        unimplemented_audiences = set(audience) - set(explanations.keys())
        for aud in unimplemented_audiences:
            explanations[aud] = 'No explanation available for this audience type.'
        
        return base_explanation | explanations 

    def create_visualization(
        self,
        patient_ctx: PatientContext,
        safety_assessment: SafetyAssessment,
        final_decision: FinalTriageDecision,
        routing_decision: Optional[RoutingDecision] = None,
        specialty_results: Optional[Dict[SpecialtyType, Dict[str, Any]]] = None,
    ) -> None:
        """
        Renders a beautiful terminal dashboard and returns the raw 
        string for logging purposes.
        """
        # We wrap this in a try-block to ensure debugging never crashes the main flow
        try:
            self.viz.render_all(
                patient=patient_ctx,
                safety=safety_assessment,
                routing=routing_decision,
                specialties=specialty_results,
                triage=final_decision
            )
        except Exception as e:
            print(f"Visualization Error: {e}")
   
    def _build_explanation_prompt(self, **kwargs) -> str:
        """Helper to construct the prompt from keyword arguments."""
        # Using .get() to safely handle optional arguments
        patient_ctx = kwargs.get('patient_ctx')
        safety = kwargs.get('safety_assessment')
        routing = kwargs.get('routing_decision')
        specialties = kwargs.get('specialty_results')
        final = kwargs.get('final_decision')

        prompt_parts = [
            "Generate a comprehensive triage explanation based on the following information:\n",
            "\n### PATIENT CASE",
            patient_ctx._patient_summary if patient_ctx else "N/A",
        ]
        
        if safety:
            prompt_parts.append(f"\n### SAFETY ASSESSMENT\n{safety.format_for_explanation()}")
        
        if routing:
            prompt_parts.append(f"\n### ROUTING\n{routing.format_for_explanation()}")

        if specialties:
            prompt_parts.append("\n### SPECIALTY EVALUATIONS")
            for spec, res in specialties.items():
                prompt_parts.append(f"**{spec.value}**:")
                # Safe access to dict keys
                if 'triage' in res:
                    prompt_parts.append(res['triage'].format_for_explanation())
                if 'validation' in res:
                    prompt_parts.append(res['validation'].format_for_explanation())
        
        if final:
            prompt_parts.append(f"\n### FINAL DECISION\n{final.format_for_explanation()}")
        
        prompt_parts.append("\n### TASK\nGenerate a clear, comprehensive explanation tailored to the context.")

        prompt_parts += [
            "\n### IMPORTANT:",
            "Do NOT include your reasoning process.",
            "Do NOT describe how you are constructing the answer.",
            "Output ONLY the final structured explanation."

        ]
        return "\n".join(prompt_parts)