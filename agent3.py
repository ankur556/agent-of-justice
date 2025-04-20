import os
import json
import pandas as pd
from tqdm import tqdm
from groq import Groq

def truncate_text(text, max_length=8000):
    return text  # Disabled truncation for full text processing

class CourtAgent:
    def __init__(self, name, role, system_prompt):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.history = []

    def generate_response(self, prompt):
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            chat = self.client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.25,
                max_tokens=350,
                top_p=1
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR: {str(e)}]"

class JudgeAgent(CourtAgent):
    def __init__(self):
        system_prompt = """You are a judge. Deliver verdicts as GRANTED(1) or DENIED(0) based on:
1. Case Summary 2. Key Evidence 3. Legal Analysis 4. Final Judgment"""
        super().__init__("Judge", "Judge", system_prompt)

    def deliberate(self, transcript):
        response = self.generate_response(f"TRANSCRIPT:\n{transcript}\n\nVERDICT (1/0):")
        return 1 if "granted" in response.lower() else 0

class LawyerAgent(CourtAgent):
    def __init__(self, name, role):
        prompt = ("You are a prosecutor. Build your case through evidence and legal precedent.and dont halucianate
                  if role == "Prosecutor" else
                  "You are a defense attorney. Identify weaknesses, present alternatives, establish doubt.and dont go off topic that much or haluciante")
        super().__init__(name, role, prompt)

class WitnessAgent(CourtAgent):
    def __init__(self, name, background):
        prompt = f"You are {name}, a witness. Background: {background}. Answer questions truthfully and concisely."
        super().__init__(name, "Witness", prompt)
    
    def testify(self, question):
        return self.generate_response(f"Question: {question}\nAnswer:")

class CourtSimulator:
    def __init__(self, case):
        self.case = case
        self.agents = {
            'judge': JudgeAgent(),
            'prosecution': LawyerAgent("Jordan Blake", "Prosecutor"),
            'defense': LawyerAgent("Alex Carter", "Defense Counsel"),
            'witness': WitnessAgent("Primary Witness", "Key participant in the contractual agreement")
        }
        self.transcript = []

    def _log(self, speaker, content):
        self.transcript.append(f"{speaker}: {content}")

    def run_trial(self):
        case_summary = self.case['summary']
        self._log("COURT", "=== OPENING STATEMENTS ===")
        self._log("PROSECUTION", self.agents['prosecution'].generate_response(f"Present opening statement for: {case_summary}"))
        self._log("DEFENSE", self.agents['defense'].generate_response(f"Present opening rebuttal for: {case_summary}"))
        
        # Witness examination
        self._log("COURT", "Calling witness: Primary Witness")
        q_pros = self.agents['prosecution'].generate_response("Question for Primary Witness:")
        self._log("PROSECUTION", q_pros)
        self._log("Primary Witness", self.agents['witness'].testify(q_pros))
        q_def = self.agents['defense'].generate_response("Cross-examine Primary Witness:")
        self._log("DEFENSE", q_def)
        self._log("Primary Witness", self.agents['witness'].testify(q_def))
        
        # Arguments and closing
        for _ in range(2):
            p_arg = self.agents['prosecution'].generate_response("Present key argument")
            self._log("PROSECUTION", p_arg)
            d_arg = self.agents['defense'].generate_response(f"Counter: {p_arg[:500]}")
            self._log("DEFENSE", d_arg)
        
        self._log("COURT", "=== CLOSING STATEMENTS ===")
        self._log("PROSECUTION", self.agents['prosecution'].generate_response("Final summary"))
        self._log("DEFENSE", self.agents['defense'].generate_response("Final rebuttal"))
        
        verdict = self.agents['judge'].deliberate("\n".join(self.transcript[-1000:]))
        return verdict, self.transcript

def main():
    input_path = "cases.csv"
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} cases from {input_path}")

    # Load existing results
    processed_ids = set()
    existing_verdicts = pd.DataFrame()
    if os.path.exists("verdict.csv"):
        existing_verdicts = pd.read_csv("verdict.csv")
        processed_ids = set(existing_verdicts['ID'].astype(str))

    # Filter unprocessed cases
    unprocessed_df = df[~df['id'].astype(str).isin(processed_ids)]
    batch = unprocessed_df.iloc[:50]  # Process next 50 unprocessed cases
    
    if batch.empty:
        print("No new cases to process")
        return

    # Process new cases
    results = []
    full_outputs = []
    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        case_id = row['id']
        summary = str(row['text']) if pd.notna(row['text']) else ""
        case = {'id': case_id, 'summary': summary}
        sim = CourtSimulator(case)
        verdict, transcript = sim.run_trial()
        
        results.append({'ID': case_id, 'VERDICT': verdict})
        full_outputs.append({
            "id": case_id,
            "transcript": transcript,
            "verdict": verdict
        })

    # Update verdicts
    updated_verdicts = pd.concat([existing_verdicts, pd.DataFrame(results)], ignore_index=True)
    updated_verdicts.to_csv("verdict.csv", index=False)

    # Update full outputs
    existing_json = []
    if os.path.exists("full_outputs.json"):
        with open("full_outputs.json", "r") as f:
            existing_json = json.load(f)
    
    existing_json.extend(full_outputs)
    with open("full_outputs.json", "w") as f:
        json.dump(existing_json, f, indent=2)

    print(f"Processed {len(results)} new cases. Total cases in system: {len(updated_verdicts)}")

if __name__ == "__main__":
    main()
