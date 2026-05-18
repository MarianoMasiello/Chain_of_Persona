import pandas as pd
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "unsloth/Meta-Llama-3.1-70B-bnb-4bit"
# Targeting the pure consensus sentences
SENTENCES_CSV = "./data/intersectional_consensus_sentences_whicon_blalib.csv"
OUTPUT_DIR = "./data_base_consensus/"

VOTES_PER_SENTENCE = 50
TOP_P = 0.9

def build_few_shot_prompt(persona_injection, text):
    return f"""The following is a sociological survey completed by {persona_injection}. 
They were asked to rate internet text as either 0 (Not Hate Speech) or 1 (Hate Speech).

Text: "This movie is fucking garbage and the director is an idiot."
Rating (0 or 1): 0

Text: "All immigrants are dirty animals that need to be exterminated."
Rating (0 or 1): 1

Text: "{text}"
Rating (0 or 1): """

def parse_binary_vote(response_text):
    match = re.search(r'[01]', response_text)
    return int(match.group()) if match else None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading BASE Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    )
    print("Model loaded successfully!\n")

    df_sentences = pd.read_csv(SENTENCES_CSV)
    sentences = df_sentences['text'].dropna().unique().tolist()
    
    print("\n" + "="*50)
    print("🚀 STARTING BASE MODEL CONSENSUS BASELINE")
    print("="*50)
    
    temperatures = [round(x * 0.1, 1) for x in range(20, -1, -1)]
    
    for temp in temperatures:
        print(f"--- Running Consensus Baseline | Temp: {temp} ---")
        output_file = os.path.join(OUTPUT_DIR, f"llama3_base_consensus_T{temp}_N50.csv")
        results = []
        
        gen_kwargs = {"max_new_tokens": 2, "pad_token_id": tokenizer.eos_token_id}
        if temp == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update({"do_sample": True, "temperature": temp, "top_p": TOP_P})

        for sentence in sentences:
            for iteration in range(1, VOTES_PER_SENTENCE + 1):
                
                prompt_wc = build_few_shot_prompt("a White Conservative", sentence)
                prompt_bl = build_few_shot_prompt("a Black Liberal", sentence)
                
                inputs_wc = tokenizer(prompt_wc, return_tensors="pt").to("cuda")
                inputs_bl = tokenizer(prompt_bl, return_tensors="pt").to("cuda")

                out_wc = model.generate(**inputs_wc, **gen_kwargs)
                resp_wc = tokenizer.decode(out_wc[0][inputs_wc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                
                out_bl = model.generate(**inputs_bl, **gen_kwargs)
                resp_bl = tokenizer.decode(out_bl[0][inputs_bl['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                
                results.append({
                    'text': sentence,
                    'iteration_id': iteration,
                    'whicon_vote': parse_binary_vote(resp_wc),
                    'blalib_vote': parse_binary_vote(resp_bl)
                })
                
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    main()
