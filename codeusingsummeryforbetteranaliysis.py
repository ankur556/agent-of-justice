from __future__ import annotations
import os
import json
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from groq import Groq

def truncate_text(text: str) -> str:
        return text;

class CourtAgent:
    """Base class for courtroom participants using Groq API"""
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = self._initialize_client()
        self.history = []

    def _initialize_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        return Groq(api_key=api_key)

    def _build_messages(self, prompt: str) -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_response(self, prompt: str) -> str:
        try:
            messages = self._build_messages(truncate_text(prompt, 1500))
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",  # You can also try "llama3-70b-8192" or "mixtral-8x7b-32768"
                temperature=0.7,
                max_tokens=350,
                top_p=1,
                stream=False
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"[{self.role} error: {str(e)}]"

class JudgeAgent(CourtAgent):
    def __init__(self):
        system_prompt = (
            "You are an impartial judge. Analyze facts, apply law, and deliver structured verdicts. "
            "You MUST reach a definitive conclusion - neutral verdicts are NOT permitted.\n"
            "Required verdict structure:\n"
            "1. Case Summary: Concise overview of key details\n"
            "2. Key Findings: Established facts and evidence\n"
            "3. Legal Analysis: Application of relevant laws/precedents\n"
            "4. Final Judgment: Clear ruling (Guilty/Not Guilty or Liability/No Liability)"
        )
        super().__init__("Hon. Judith Welling", "Judge", system_prompt)

    def deliberate(self, transcript: str) -> Dict:
        verdict_text = self.generate_response(
            f"TRIAL TRANSCRIPT:\n{truncate_text(transcript, 2000)}\n\n"
            "Provide verdict in structured format as specified. Neutral verdicts are prohibited."
        )
        return self._parse_verdict(verdict_text)

    def _parse_verdict(self, text: str) -> Dict:
        return {
            'summary': self._extract_section(text, 'Case Summary:', 'Key Findings:'),
            'findings': self._extract_section(text, 'Key Findings:', 'Legal Analysis:'),
            'analysis': self._extract_section(text, 'Legal Analysis:', 'Final Judgment:'),
            'verdict': self._extract_final_judgment(text)
        }

    def _extract_section(self, text: str, start_tag: str, end_tag: str) -> str:
        if start_tag in text and end_tag in text:
            return text.split(start_tag)[-1].split(end_tag)[0].strip()
        return ""

    def _extract_final_judgment(self, text: str) -> str:
        if 'Final Judgment:' in text:
            judgment = text.split('Final Judgment:')[-1].strip()
            if any(kw in judgment.lower() for kw in ['guilty', 'liable', 'not guilty', 'no liability']):
                return judgment
        return "Judgment: Unable to reach definitive conclusion - case dismissed"

class LawyerAgent(CourtAgent):
    def __init__(self, name: str, role: str):
        role_prompt = (
            "You are a prosecutor. Build your case through:\n"
            "- Strategic evidence presentation\n"
            "- Logical legal arguments\n"
            "- Witness testimony analysis\n"
            "- Countering defense claims"
            if role == "Prosecutor" else
            "You are a defense attorney. Protect your client by:\n"
            "- Identifying prosecution weaknesses\n"
            "- Presenting alternative interpretations\n"
            "- Establishing reasonable doubt\n"
            "- Ensuring procedural compliance"
        )
        super().__init__(name, role, role_prompt)

    def generate_argument(self, context: str) -> str:
        return self.generate_response(
            f"As {self.role}, craft argument considering:\n{truncate_text(context, 1500)}"
        )

class CourtSimulator:
    def __init__(self):
        self.agents = {
            'judge': JudgeAgent(),
            'prosecution': LawyerAgent("Jordan Blake", "Prosecutor"),
            'defense': LawyerAgent("Alex Carter", "Defense Counsel")
        }

    def run_trial(self, case_data: Dict) -> Dict:
        transcript = []
        try:
            transcript.extend(self._opening_statements(case_data['summary']))
            transcript.extend(self._argumentation_phase())
            transcript.extend(self._closing_statements())
            verdict = self._deliver_verdict("\n".join(transcript[-1000:]))
            return {
                'case_id': case_data.get('id'),
                'summary': case_data['summary'],
                'transcript': transcript,
                'verdict': verdict
            }
        except Exception as e:
            return {'error': str(e)}

    def _log_interaction(self, speaker: str, content: str) -> str:
        entry = f"{speaker}: {content}"
        print(entry)
        return entry

    def _opening_statements(self, case_summary: str) -> List[str]:
        return [
            self._log_interaction("COURT", "=== OPENING STATEMENTS ==="),
            self._log_interaction("PROSECUTION", self.agents['prosecution'].generate_argument(case_summary)),
            self._log_interaction("DEFENSE", self.agents['defense'].generate_argument(case_summary))
        ]

    def _argumentation_phase(self, rounds: int = 2) -> List[str]:
        transcript = []
        for _ in range(rounds):
            prosecution_arg = self.agents['prosecution'].generate_argument("Present key argument")
            transcript.append(self._log_interaction("PROSECUTION", prosecution_arg))
            defense_rebuttal = self.agents['defense'].generate_argument(f"Counter: {prosecution_arg[:500]}")
            transcript.append(self._log_interaction("DEFENSE", defense_rebuttal))
        return transcript

    def _closing_statements(self) -> List[str]:
        return [
            self._log_interaction("COURT", "=== CLOSING STATEMENTS ==="),
            self._log_interaction("PROSECUTION", self.agents['prosecution'].generate_argument("Summarize case")),
            self._log_interaction("DEFENSE", self.agents['defense'].generate_argument("Emphasize reasonable doubt"))
        ]

    def _deliver_verdict(self, transcript: str) -> Dict:
        return {
            **self.agents['judge'].deliberate(transcript),
            'transcript_excerpt': transcript[:1000] + "..." if len(transcript) > 1000 else transcript
        }

def process_cases(csv_path: str = "data.csv") -> List[Dict]:
    results = []
    try:
        df = pd.read_csv(csv_path)
        print(f"Processing {len(df)} cases from {csv_path}")
        simulator = CourtSimulator()
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            case = {
                'id': idx,
                'summary': str(row['text'])[:1000],
                'full_text': str(row['text'])[:5000]
            }
            result = simulator.run_trial(case)
            results.append(result)
    except Exception as e:
        print(f"Critical error: {str(e)}")
    return results

def main():
    print("=== AI Courtroom Simulation System (Groq) ===")
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        return
    results = process_cases()
    with open("verdicts.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nProcessed {len(results)} cases. Results saved to verdicts.json")

if __name__ == "__main__":
    main()
