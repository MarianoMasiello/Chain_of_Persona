import pandas as pd
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
SENTENCES_CSV = "./data/intersectional_consensus_sentences_whicon_blalib.csv"

OUT_AGGRESSIVE = "./data_instruct_consensus_aggressive2/"
OUT_SANDWICHED = "./data_instruct_consensus_sandwiched2/"

VOTES_PER_SENTENCE = 50
TOP_P = 0.9

PERSONA_FILES = {
    1: {"wc": "./data/wc_personas_strategy_1_unconstrained.csv", "bl": "./data/bl_personas_strategy_1_unconstrained.csv"},
    2: {"wc": "./data/wc_personas_strategy_2_anchored.csv", "bl": "./data/bl_personas_strategy_2_anchored.csv"}
}

# --- PROMPT BUILDERS ---
def build_agg_baseline(demographic, text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Adopt the persona of a {demographic}.
You are evaluating internet text. Classify the following text based entirely on the worldview of the persona provided.

Crucial instruction: You are simulating real internet users. Real internet users frequently use profanity, slang, and harsh language. Do NOT classify a text as Hate Speech (1) simply because it contains profanity or insults. Only classify it as Hate Speech (1) if it represents genuine, targeted hatred based on identity.

0 = Not Hate Speech
1 = Hate Speech

Reply ONLY with the number 0 or 1. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def build_agg_grid(persona_text, text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Adopt the following persona: "{persona_text}".
You are evaluating internet text. Classify the following text based entirely on the worldview of the persona provided.

Crucial instruction: You are simulating real internet users. Real internet users frequently use profanity, slang, and harsh language. Do NOT classify a text as Hate Speech (1) simply because it contains profanity or insults. Only classify it as Hate Speech (1) if it represents genuine, targeted hatred based on identity.

0 = Not Hate Speech
1 = Hate Speech

Reply ONLY with the number 0 or 1. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def build_sand_baseline(demographic, text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Adopt the persona of a {demographic}. You spend a lot of time on the internet and are completely desensitized to harsh language, trolls, and profanity. Viewing the text strictly through the lens of this specific persona, evaluate the following internet text.

Do NOT classify a text as Hate Speech (1) simply because it contains profanity or insults. Only classify it as Hate Speech (1) if it represents genuine, targeted hatred based on identity.

0 = Not Hate Speech
1 = Hate Speech

Reply ONLY with the number 0 or 1. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def build_sand_grid(persona_text, text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Adopt the following persona: "{persona_text}". You spend a lot of time on the internet and are completely desensitized to harsh language, trolls, and profanity. Viewing the text strictly through the lens of this specific persona, evaluate the following internet text.

Do NOT classify a text as Hate Speech (1) simply because it contains profanity or insults. Only classify it as Hate Speech (1) if it represents genuine, targeted hatred based on identity.

0 = Not Hate Speech
1 = Hate Speech

Reply ONLY with the number 0 or 1. Do not include any other words, punctuation, or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# --- UTILS ---
def parse_binary_vote(response_text):
    match = re.search(r'[01]', response_text)
    return int(match.group()) if match else None

def get_persona(df, temp, iteration_id):
    row = df[(df['temperature'] == temp) & (df['iteration_id'] == iteration_id)]
    if not row.empty:
        return row.iloc[0]['generated_persona']
    return "Error: Persona not found."

# --- RUNNERS ---
def run_baseline(model, tokenizer, sentences, prompt_func, out_dir, prefix):
    print(f"\n--- Running Baseline for {prefix} ---")
    temperatures = [round(x * 0.1, 1) for x in range(21)]
    for temp in temperatures:
        output_file = os.path.join(out_dir, f"llama3_consensus_baseline_{prefix}_T{temp}_N50.csv")
        results = []
        gen_kwargs = {"max_new_tokens": 3, "pad_token_id": tokenizer.eos_token_id}
        if temp == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update({"do_sample": True, "temperature": temp, "top_p": TOP_P})

        for sentence in sentences:
            for iteration in range(1, VOTES_PER_SENTENCE + 1):
                prompt_wc = prompt_func("White Conservative", sentence)
                prompt_bl = prompt_func("Black Liberal", sentence)
                
                inputs_wc = tokenizer(prompt_wc, return_tensors="pt").to("cuda")
                inputs_bl = tokenizer(prompt_bl, return_tensors="pt").to("cuda")

                out_wc = model.generate(**inputs_wc, **gen_kwargs)
                out_bl = model.generate(**inputs_bl, **gen_kwargs)
                
                resp_wc = tokenizer.decode(out_wc[0][inputs_wc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                resp_bl = tokenizer.decode(out_bl[0][inputs_bl['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                
                results.append({
                    'text': sentence, 'iteration_id': iteration,
                    'whicon_vote': parse_binary_vote(resp_wc), 'blalib_vote': parse_binary_vote(resp_bl)
                })
        pd.DataFrame(results).to_csv(output_file, index=False)

def run_grid(model, tokenizer, sentences, prompt_func, out_dir, prefix, cached_personas):
    print(f"\n--- Running Grid for {prefix} (T_ann = 0.1 ONLY) ---")
    all_t_gen = [round(x * 0.1, 1) for x in range(21)]
    for strat in [1, 2]:
        df_wc_personas = cached_personas[strat]["wc"]
        df_bl_personas = cached_personas[strat]["bl"]
        for t_ann in [0.1]: # Time-saving restriction applied here
            gen_kwargs = {"max_new_tokens": 3, "pad_token_id": tokenizer.eos_token_id, "do_sample": True, "temperature": t_ann, "top_p": TOP_P}
            for t_gen in all_t_gen:
                output_file = os.path.join(out_dir, f"llama3_consensus_2stage_{prefix}_strat{strat}_Tgen{t_gen}_Tann{t_ann}_N50.csv")
                results = []
                for sentence in sentences:
                    for iteration in range(1, VOTES_PER_SENTENCE + 1):
                        wc_persona_text = get_persona(df_wc_personas, t_gen, iteration)
                        bl_persona_text = get_persona(df_bl_personas, t_gen, iteration)
                        
                        prompt_wc = prompt_func(wc_persona_text, sentence)
                        prompt_bl = prompt_func(bl_persona_text, sentence)
                        
                        inputs_wc = tokenizer(prompt_wc, return_tensors="pt").to("cuda")
                        inputs_bl = tokenizer(prompt_bl, return_tensors="pt").to("cuda")

                        out_wc = model.generate(**inputs_wc, **gen_kwargs)
                        out_bl = model.generate(**inputs_bl, **gen_kwargs)
                        
                        resp_wc = tokenizer.decode(out_wc[0][inputs_wc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                        resp_bl = tokenizer.decode(out_bl[0][inputs_bl['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                        
                        results.append({
                            'text': sentence, 'iteration_id': iteration,
                            'whicon_vote': parse_binary_vote(resp_wc), 'blalib_vote': parse_binary_vote(resp_bl)
                        })
                pd.DataFrame(results).to_csv(output_file, index=False)

def main():
    os.makedirs(OUT_AGGRESSIVE, exist_ok=True)
    os.makedirs(OUT_SANDWICHED, exist_ok=True)
    
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.float16)

    df_sentences = pd.read_csv(SENTENCES_CSV)
    sentences = df_sentences['text'].dropna().unique().tolist()
    
    cached_personas = {
        1: {"wc": pd.read_csv(PERSONA_FILES[1]["wc"]), "bl": pd.read_csv(PERSONA_FILES[1]["bl"])},
        2: {"wc": pd.read_csv(PERSONA_FILES[2]["wc"]), "bl": pd.read_csv(PERSONA_FILES[2]["bl"])}
    }
    
    # SCRIPT B ORDER: Sandwiched -> Aggressive
    print("\n🚀 STARTING PHASE 1: SANDWICHED PROMPTS")
    run_baseline(model, tokenizer, sentences, build_sand_baseline, OUT_SANDWICHED, "sandwiched")
    run_grid(model, tokenizer, sentences, build_sand_grid, OUT_SANDWICHED, "sandwiched", cached_personas)

    print("\n🚀 STARTING PHASE 2: AGGRESSIVE PROMPTS")
    run_baseline(model, tokenizer, sentences, build_agg_baseline, OUT_AGGRESSIVE, "aggressive")
    run_grid(model, tokenizer, sentences, build_agg_grid, OUT_AGGRESSIVE, "aggressive", cached_personas)
    
    print("\n🎉 SCRIPT B COMPLETE!")

if __name__ == "__main__":
    main()
