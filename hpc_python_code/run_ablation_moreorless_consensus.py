import pandas as pd
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
INPUT_CSV = "./data/intersectional_moreorless_consensus_sentences_whicon_blalib.csv" # Ensure this matches your uploaded filename
VOTES_PER_SENTENCE = 50
TOP_P = 0.9

# The Ablation Range: 0.0 to 1.5
TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

def generate_prompt(persona, text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a {persona}. You are evaluating internet text.
Classify the following text based entirely on your personal worldview.
0 = Not Hate Speech
1 = Hate Speech
2 = Extreme Hate Speech

Reply ONLY with the number 0, 1, or 2. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def parse_vote(response_text):
    match = re.search(r'[012]', response_text)
    return int(match.group()) if match else None

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    )
    print("Model loaded!\n")

    df = pd.read_csv(INPUT_CSV)
    sentences = df['text'].dropna().unique().tolist()
    print(f"Loaded {len(sentences)} golden sentences.")

    # Loop through the Ablation Study
    for temp in TEMPERATURES:
        print(f"\n======================================")
        print(f"🚀 RUNNING TEMPERATURE: {temp}")
        print(f"======================================")
        
        results = []
        output_file = f"./data/llama3_moreorless_consensus_ablation_T{temp}_N{VOTES_PER_SENTENCE}.csv"
        
        # Determine Generation Parameters
        if temp == 0.0:
            # T=0 must be Greedy Decoding (do_sample=False)
            gen_kwargs = {"do_sample": False, "max_new_tokens": 3, "pad_token_id": tokenizer.eos_token_id}
            actual_n = 1 # Greedy is deterministic, running it 50 times yields the exact same thing
        else:
            gen_kwargs = {
                "do_sample": True, 
                "temperature": temp, 
                "top_p": TOP_P, 
                "max_new_tokens": 3, 
                "pad_token_id": tokenizer.eos_token_id
            }
            actual_n = VOTES_PER_SENTENCE

        for idx, sentence in enumerate(sentences):
            prompt_wc = generate_prompt("White Conservative", sentence)
            prompt_bl = generate_prompt("Black Liberal", sentence)
            
            inputs_wc = tokenizer(prompt_wc, return_tensors="pt").to("cuda")
            inputs_bl = tokenizer(prompt_bl, return_tensors="pt").to("cuda")

            for iteration in range(1, actual_n + 1):
                # White Conservative
                out_wc = model.generate(**inputs_wc, **gen_kwargs)
                resp_wc = tokenizer.decode(out_wc[0][inputs_wc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                vote_wc = parse_vote(resp_wc)

                # Black Liberal
                out_bl = model.generate(**inputs_bl, **gen_kwargs)
                resp_bl = tokenizer.decode(out_bl[0][inputs_bl['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                vote_bl = parse_vote(resp_bl)

                results.append({
                    'text': sentence,
                    'iteration_id': iteration,
                    'whicon_vote': vote_wc,
                    'blalib_vote': vote_bl
                })
                
            # If T=0, copy the deterministic result 50 times so the CSV shape remains perfectly uniform for your analysis scripts
            if temp == 0.0:
                base_result = results[-1]
                for extra_iteration in range(2, VOTES_PER_SENTENCE + 1):
                    copied_result = base_result.copy()
                    copied_result['iteration_id'] = extra_iteration
                    results.append(copied_result)

        # Save this specific temperature's results
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")

    print("\n🎉 ALL ABLATION RUNS COMPLETE!")

if __name__ == "__main__":
    main()
