from __future__ import annotations
import os
import json
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
import numpy as np

class LegalAgent:
    def __init__(self,
                 name: str,
                 role: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct",
                 experience: List[Dict] = None):
        self.name = name
        self.role = role
        self.model = model
        self.experience = experience or []
        self.client = InferenceClient(model, token=os.getenv("HF_API_TOKEN"))

    def evolve(self, case_outcome: float):
        self.model = self._mutate_model(case_outcome)

    def _mutate_model(self, fitness: float) -> str:
        models = [
            "microsoft/Phi-3-mini-4k-instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/gemma-7b-it"
        ]
        return np.random.choice(models, p=[0.6, 0.3, 0.1]) if fitness < 0.5 else self.model

class LawyerAgent(LegalAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.argument_strategy = {
            'precedent_weight': 0.7,
            'emotional_appeal': 0.2,
            'factual_focus': 0.8
        }

    def generate_argument(self, case_context: str) -> str:
        prompt = f"""As {self.role} {self.name}, craft legal argument considering:
        - Relevant precedents from {self.experience}
        - Case facts: {case_context}
        - Optimal strategy weights: {self.argument_strategy}"""
        return self.client.text_generation(prompt, max_new_tokens=512, temperature=0.7)

class JudgeAgent(LegalAgent):
    def __init__(self, legal_kb: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.legal_kb = legal_kb

    def deliberate(self, transcript: str) -> Dict:
        analysis_prompt = f"""Analyze trial transcript using legal standards:
        TRANSCRIPT: {transcript}
        RELEVANT PRECEDENTS: {json.dumps(self.legal_kb)}

        Produce JSON verdict with:
        - summary: Case overview
        - findings: Key factual determinations
        - application: Legal standard application
        - verdict: Final judgment
        - reasoning: Detailed legal analysis"""
        return json.loads(self.client.text_generation(analysis_prompt, max_new_tokens=1024))

class CourtSimulator:
    def __init__(self, case: Dict):
        self.case = case
        self.agents = {
            'judge': JudgeAgent(name="Hon. Judith Welling", role="Judge",
                              legal_kb=self._load_precedents()),
            'prosecution': LawyerAgent(name="Jordan Blake", role="Prosecutor"),
            'defense': LawyerAgent(name="Alex Carter", role="Defense Counsel")
        }
        self.transcript = []

    def _load_precedents(self) -> List[Dict]:
        return [{
            "citation": "State v. Doe, 2023",
            "holding": "Circumstantial evidence sufficient for trade secret theft",
            "application": "Requires proof of access and substantial similarity"
        }]

    def run_trial_phase(self, phase: str):
        phases = {
            'open': self._opening_statements,
            'close': self._closing_arguments,
            # 'exam': self._witness_examination,  # Add if you implement witnesses
        }
        if phase in phases:
            phases[phase]()

    def _log_interaction(self, speaker: str, content: str):
        entry = f"{speaker.upper()}: {content}"
        self.transcript.append(entry)
        print(entry)

    def _opening_statements(self):
        prosecution_open = self.agents['prosecution'].generate_argument(self.case['summary'])
        self._log_interaction('Prosecution', prosecution_open)

        defense_open = self.agents['defense'].generate_argument(self.case['summary'])
        self._log_interaction('Defense', defense_open)

    def adversarial_exchange(self, rounds: int = 3):
        for _ in range(rounds):
            prosecution_arg = self.agents['prosecution'].generate_argument(
                f"Rebuttal to: {self.transcript[-1]}"
            )
            self._log_interaction('Prosecution', prosecution_arg)

            defense_arg = self.agents['defense'].generate_argument(
                f"Counter: {prosecution_arg}"
            )
            self._log_interaction('Defense', defense_arg)
            self._evaluate_performance()

    def _evaluate_performance(self):
        prosecution_score = np.random.rand()
        defense_score = np.random.rand()
        self.agents['prosecution'].evolve(prosecution_score)
        self.agents['defense'].evolve(defense_score)

    def deliver_verdict(self) -> Dict:
        full_transcript = "\n".join(self.transcript)
        return self.agents['judge'].deliberate(full_transcript)

# Example Usage for paste.txt
if __name__ == "__main__" or True:  # So it runs in Jupyter
    # Read your plain text file
    with open("paste.txt", "r", encoding="utf-8") as f:
        case_text = f.read()

    # Structure the case for the simulator
    case = {
        "summary": case_text[:1000],  # Use the first 1000 chars as summary (adjust as needed)
        "full_text": case_text
        # Add more fields if needed
    }

    simulator = CourtSimulator(case)
    simulator.run_trial_phase('open')
    simulator.adversarial_exchange(rounds=3)
    simulator.run_trial_phase('close')
    verdict = simulator.deliver_verdict()
    print("\nFINAL VERDICT:")
    print(json.dumps(verdict, indent=2))
