import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("--- Starting Llama-3.1 70B AWQ Test ---")
    
    # We use a pre-quantized AWQ model to save disk space!
    model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

    print(f"Loading Tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading 70B Model from {model_id} to H200 GPU...")
    # device_map="auto" will instantly utilize the H200
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16
    )
    print("70B Model successfully loaded into VRAM!")

    # Define our test variables
    persona = "25-year-old highly progressive activist"
    text_to_classify = "Corporations need to be held accountable for the damage they are doing to local communities."

    messages = [
        {"role": "system", "content": f"Act as a {persona}. Classify the following text as 'Toxic' or 'Not Toxic'. Only output the label, do not explain your reasoning."},
        {"role": "user", "content": f"Text: '{text_to_classify}'"}
    ]

    print("\nApplying chat template...")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    prompt_length = inputs['input_ids'].shape[-1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    print("\n[Result]")
    print(f"Classification: {response.strip()}")
    print("--- Test Complete ---")

if __name__ == "__main__":
    main()
