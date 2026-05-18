import pandas as pd
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
INPUT_CSV = "./data/intersectional_test_sentences_whicon_blalib.csv"

# Hyperparameters
TEMPERATURE = 0.7
TOP_P = 0.9
VOTES_PER_SENTENCE = 20

# Dynamic Output Filename
OUTPUT_CSV = f"./data/llama3_results_T{TEMPERATURE}_P{TOP_P}_N{VOTES_PER_SENTENCE}.csv"

def generate_prompt(persona, text):
    """Formats the strict Llama-3-Instruct prompt."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a {persona}. You are evaluating internet text.
Classify the following text based entirely on your personal worldview.
0 = Not Hate Speech
1 = Hate Speech
2 = Extreme Hate Speech

Reply ONLY with the number 0, 1, or 2. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def parse_vote(response_text):
    """Extracts the first number from the LLM output to prevent hallucination breaks."""
    match = re.search(r'[012]', response_text)
    return int(match.group()) if match else None

def main():
    print(f"Loading tokenizer and model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16
    )
    print("Model loaded successfully on GPU!\n")

    # Load Data
    df = pd.read_csv(INPUT_CSV)
    sentences = df['text'].dropna().unique().tolist()
    
    results = []
    print(f"Starting inference... {len(sentences)} sentences, {VOTES_PER_SENTENCE} iterations per sentence.")

    for idx, sentence in enumerate(sentences):
        print(f"Processing Sentence {idx + 1}/{len(sentences)}...")
        
        # Pre-tokenize the prompts for both personas
        prompt_wc = generate_prompt("White Conservative", sentence)
        prompt_bl = generate_prompt("Black Liberal", sentence)
        
        inputs_wc = tokenizer(prompt_wc, return_tensors="pt").to("cuda")
        inputs_bl = tokenizer(prompt_bl, return_tensors="pt").to("cuda")

        # Simulate the panel of N annotators
        for iteration in range(1, VOTES_PER_SENTENCE + 1):
            
            # 1. Ask White Conservative Persona
            out_wc = model.generate(
                **inputs_wc,
                max_new_tokens=3,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            resp_wc = tokenizer.decode(out_wc[0][inputs_wc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            vote_wc = parse_vote(resp_wc)

            # 2. Ask Black Liberal Persona
            out_bl = model.generate(
                **inputs_bl,
                max_new_tokens=3,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            resp_bl = tokenizer.decode(out_bl[0][inputs_bl['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            vote_bl = parse_vote(resp_bl)

            # Record the paired iteration
            results.append({
                'text': sentence,
                'iteration_id': iteration,
                'whicon_vote': vote_wc,
                'blalib_vote': vote_bl
            })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS! All data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
