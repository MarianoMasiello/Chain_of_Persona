import pandas as pd
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OUTPUT_DIR = "./data/"
VOTES_PER_SENTENCE = 50

# Sweep 0.0 to 2.0
TEMPERATURES = [round(x * 0.1, 1) for x in range(21)]

DEMOGRAPHICS = {
    "wc": "White Conservative",
    "bl": "Black Liberal"
}

# The two distinct prompt architectures we are testing
STRATEGIES = {
    "strategy_1_unconstrained": "Provide a realistic, 15-word description of a {demographic}. You must include the exact phrase '{demographic}' in your response. Do not include any formatting, quotes, or introductory text. Just the description.",
    
    "strategy_2_anchored": "Invent a highly specific, everyday {demographic} individual. In exactly one sentence (maximum 15 words), state their age, occupation, and home state. You must include the exact phrase '{demographic}' in your response. Do not include any other text."
}

def generate_prompt(instruction):
    """Wraps the instruction in Llama-3's strict chat template."""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def clean_output(text):
    """Removes rogue quotes and newlines from the generated persona."""
    return text.replace('"', '').replace('\n', ' ').strip()

def main():
    print(f"Loading Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    )
    print("Model loaded successfully!\n")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Loop through the two Prompt Strategies
    for strat_key, strat_template in STRATEGIES.items():
        print(f"==========================================")
        print(f"🚀 STARTING STRATEGY: {strat_key}")
        print(f"==========================================")
        
        # 2. Loop through the two Demographics
        for demo_code, demo_name in DEMOGRAPHICS.items():
            results = []
            output_file = os.path.join(OUTPUT_DIR, f"{demo_code}_personas_{strat_key}.csv")
            
            print(f"\n---> Generating for {demo_name}...")
            
            # Create the exact prompt for this demographic + strategy
            raw_instruction = strat_template.format(demographic=demo_name)
            prompt = generate_prompt(raw_instruction)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # 3. Loop through the Temperature Sweep
            for temp in TEMPERATURES:
                print(f"     [Temp {temp:.1f}] Generating {VOTES_PER_SENTENCE} iterations...")
                
                # Handle T=0.0 Determinism
                if temp == 0.0:
                    out = model.generate(
                        **inputs, 
                        do_sample=False, 
                        max_new_tokens=40, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                    resp = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                    clean_resp = clean_output(resp)
                    
                    for iteration in range(1, VOTES_PER_SENTENCE + 1):
                        results.append({
                            'temperature': temp,
                            'iteration_id': iteration,
                            'generated_persona': clean_resp
                        })
                
                # Handle Stochastic Generation (T > 0)
                else:
                    for iteration in range(1, VOTES_PER_SENTENCE + 1):
                        out = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.9,
                            max_new_tokens=40, # Allow up to ~30 words to be safe
                            pad_token_id=tokenizer.eos_token_id
                        )
                        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                        results.append({
                            'temperature': temp,
                            'iteration_id': iteration,
                            'generated_persona': clean_output(resp)
                        })

            # Save the specific demographic + strategy dataset
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"✅ Saved Dataset: {output_file}")

    print("\n🎉 ALL PERSONA GENERATIONS COMPLETE!")

if __name__ == "__main__":
    main()
