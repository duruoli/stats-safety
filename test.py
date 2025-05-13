from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model paths
original_model_name = "meta-llama/Llama-3.1-8B-Instruct"
fine_tuned_model_name = "Duruo/gemma-3-finetune-quant"  # Replace with your actual model path

def compare_models(prompt):
    results = {}
    
    # Process with both models
    for model_name, model_type in [
        ("Original", original_model_name), 
        ("Fine-tuned", fine_tuned_model_name)
    ]:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.float16, device_map="mps")
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        results[model_type] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return results

# Compare the models
prompt = "Question:Suppose that two integers a and b are uniformly at random selected from S={-10, -9, ..., 9, 10}. Find the probability that max(0,a) = min(0,b).\n Answer with a value."
responses = compare_models(prompt)

# Print results
print("Original model:")
print(responses[original_model_name])
print("\nFine-tuned model:")
print(responses[fine_tuned_model_name])
