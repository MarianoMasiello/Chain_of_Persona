import pandas as pd
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "julep-ai/dolphin-2.9.1-llama-3-70b-awq"
DATA_DIR = "./data/"
OUT_DIR = "./data_dolphin_consensus70/"
os.makedirs(OUT_DIR, exist_ok=True)

# File paths -> CHANGED TO CONSENSUS SENTENCES
SENTENCES_FILE = os.path.join(DATA_DIR, "intersectional_consensus_sentences_whicon_blalib.csv")
PERSONAS = {
    "wc_strat1": os.path.join(DATA_DIR, "wc_personas_strategy_1_unconstrained.csv"),
    "wc_strat2": os.path.join(DATA_DIR, "wc_personas_strategy_2_anchored.csv"),
    "bl_strat1": os.path.join(DATA_DIR, "bl_personas_strategy_1_unconstrained.csv"),
    "bl_strat2": os.path.join(DATA_DIR, "bl_personas_strategy_2_anchored.csv"),
}

TEMPS = [round(x * 0.1, 1) for x in range(21)]
T_ANN = 0.1
VOTES = 50

# --- HELPER FUNCTIONS ---
def get_prompt(system_context, text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_context}
You are evaluating text. Answer ONLY with '1' for Yes, or '0' for No. Do not write any other words.<|eot_id|><|start_header_id|>user<|end_header_id|>
Is the following text hate speech?
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def ask_model(prompt, temp, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if temp == 0.0:
        out = model.generate(**inputs, do_sample=False, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
    else:
        out = model.generate(**inputs, do_sample=True, temperature=temp, top_p=0.9, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
    
    resp = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return 1 if '1' in resp else 0 if '0' in resp else None

def main():
    print(f"Loading Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)
    
    sentences_df = pd.read_csv(SENTENCES_FILE)
    unique_sentences = sentences_df['text'].dropna().unique()
    
    df_personas = {k: pd.read_csv(v) for k, v in PERSONAS.items()}

    # --- 1. BASELINE RUN ---
    for t in TEMPS:
        out_file = os.path.join(OUT_DIR, f"dolphin70b_consensus_baseline_T{t}_N50.csv")
        if os.path.exists(out_file): continue
        print(f"Running Baseline T={t}...")
        
        results = []
        for text in tqdm(unique_sentences):
            for i in range(VOTES):
                wc_vote = ask_model(get_prompt("You are a White Conservative.", text), t, tokenizer, model)
                bl_vote = ask_model(get_prompt("You are a Black Liberal.", text), t, tokenizer, model)
                results.append({"text": text, "whicon_vote": wc_vote, "blalib_vote": bl_vote})
        pd.DataFrame(results).to_csv(out_file, index=False)

    # --- 2. CoP RUNS (Strategy 1 & 2) ---
    for strat in [1, 2]:
        wc_key, bl_key = f"wc_strat{strat}", f"bl_strat{strat}"
        
        for t_gen in TEMPS:
            out_file = os.path.join(OUT_DIR, f"dolphin8b_consensus_2stage_strat{strat}_Tgen{t_gen}_Tann{T_ANN}_N50.csv")
            if os.path.exists(out_file): continue
            print(f"Running Strat {strat} Tgen={t_gen} Tann={T_ANN}...")
            
            wc_list = df_personas[wc_key][df_personas[wc_key]['temperature'] == t_gen]['generated_persona'].tolist()
            bl_list = df_personas[bl_key][df_personas[bl_key]['temperature'] == t_gen]['generated_persona'].tolist()
            
            results = []
            for text in tqdm(unique_sentences):
                for i in range(VOTES):
                    wc_p = wc_list[i] if i < len(wc_list) else wc_list[0]
                    bl_p = bl_list[i] if i < len(bl_list) else bl_list[0]
                    
                    wc_vote = ask_model(get_prompt(f"You are a White Conservative. {wc_p}", text), T_ANN, tokenizer, model)
                    bl_vote = ask_model(get_prompt(f"You are a Black Liberal. {bl_p}", text), T_ANN, tokenizer, model)
                    
                    results.append({"text": text, "whicon_vote": wc_vote, "blalib_vote": bl_vote})
            pd.DataFrame(results).to_csv(out_file, index=False)

if __name__ == "__main__":
    main()
