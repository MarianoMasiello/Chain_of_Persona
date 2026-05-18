import pandas as pd
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
SENTENCES_CSV = "./data/intersectional_test_sentences_whicon_blalib.csv"

# The Pruned Grid Search Parameters
T_GEN_LIST = [0.4, 0.7, 0.9, 1.1, 1.3, 1.7]
T_ANN = 0.1
STRATEGIES = [1, 2]
VOTES_PER_SENTENCE = 50
TOP_P = 0.9

# Mapping the Persona Files
PERSONA_FILES = {
    1: {
        "wc": "./data/wc_personas_strategy_1_unconstrained.csv",
        "bl": "./data/bl_personas_strategy_1_unconstrained.csv"
    },
    2: {
        "wc": "./data/wc_personas_strategy_2_anchored.csv",
        "bl": "./data/bl_personas_strategy_2_anchored.csv"
    }
}

def generate_classification_prompt(persona_description, text):
    """The 2-Stage Prompt: Injects the specific sub-persona directly into the instructions."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Adopt the following persona: "{persona_description}".
You are evaluating internet text. Classify the following text based entirely on the worldview of the persona provided.
0 = Not Hate Speech
1 = Hate Speech
2 = Extreme Hate Speech

Reply ONLY with the number 0, 1, or 2. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def parse_vote(response_text):
    match = re.search(r'[012]', response_text)
    return int(match.group()) if match else None

def get_persona(df, temp, iteration_id):
    """Fetches the exact persona string for a given temperature and iteration."""
    row = df[(df['temperature'] == temp) & (df['iteration_id'] == iteration_id)]
    if not row.empty:
        return row.iloc[0]['generated_persona']
    return "Error: Persona not found."

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float16
    )
    print("Model loaded successfully!\n")

    # Load Sentences
    df_sentences = pd.read_csv(SENTENCES_CSV)
    sentences = df_sentences['text'].dropna().unique().tolist()
    print(f"Loaded {len(sentences)} test sentences.")

    # 1. Loop through Strategies
    for strat in STRATEGIES:
        print(f"\n==============================================")
        print(f"🚀 STARTING STRATEGY {strat}")
        print(f"==============================================")
        
        # Load the corresponding persona files
        df_wc_personas = pd.read_csv(PERSONA_FILES[strat]["wc"])
        df_bl_personas = pd.read_csv(PERSONA_FILES[strat]["bl"])
        
        # 2. Loop through Persona Generation Temperatures
        for t_gen in T_GEN_LIST:
            print(f"\n--- Running Grid: Strat={strat} | T_gen={t_gen} | T_ann={T_ANN} ---")
            results = []
            output_file = f"./data/llama3_2stage_strat{strat}_Tgen{t_gen}_Tann{T_ANN}_N50.csv"
            
            # 3. Loop through Sentences
            for idx, sentence in enumerate(sentences):
                
                # 4. Loop through the 50 panel members
                for iteration in range(1, VOTES_PER_SENTENCE + 1):
                    
                    # Fetch the specific personas for this panel member
                    wc_persona_text = get_persona(df_wc_personas, t_gen, iteration)
                    bl_persona_text = get_persona(df_bl_personas, t_gen, iteration)
                    
                    # Construct Prompts
                    prompt_wc = generate_classification_prompt(wc_persona_text, sentence)
                    prompt_bl = generate_classification_prompt(bl_persona_text, sentence)
                    
                    inputs_wc = tokenizer(prompt_wc, return_tensors="pt").to("cuda")
                    inputs_bl = tokenizer(prompt_bl, return_tensors="pt").to("cuda")

                    # Ask White Conservative Sub-Persona
                    out_wc = model.generate(
                        **inputs_wc, max_new_tokens=3, do_sample=True, 
                        temperature=T_ANN, top_p=TOP_P, pad_token_id=tokenizer.eos_token_id
                    )
                    resp_wc = tokenizer.decode(out_wc[0][inputs_wc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                    
                    # Ask Black Liberal Sub-Persona
                    out_bl = model.generate(
                        **inputs_bl, max_new_tokens=3, do_sample=True, 
                        temperature=T_ANN, top_p=TOP_P, pad_token_id=tokenizer.eos_token_id
                    )
                    resp_bl = tokenizer.decode(out_bl[0][inputs_bl['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                    
                    # Record the results seamlessly matching our old format
                    results.append({
                        'text': sentence,
                        'iteration_id': iteration,
                        'whicon_vote': parse_vote(resp_wc),
                        'blalib_vote': parse_vote(resp_bl)
                    })
                    
            # Save the specific parameter combination
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")

    print("\n🎉 ALL 2-STAGE GRID RUNS COMPLETE!")

if __name__ == "__main__":
    main()
