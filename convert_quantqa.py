import json
import re
import pandas as pd
from datasets import Dataset

# Load the quant_qa.json file
with open('quant_qa.json', 'r') as f:
    problems = f.read().split('``````')

# Parse the problems into MathQA format
parsed_problems = []
for problem in problems:
    if not problem.strip():
        continue
    
    # Extract problem name, question, reasoning, and answer
    match = re.search(r'Name: (.*?)\nQuestion: (.*?)\nReasoning: (.*?)\nAnswer: (.*?)$', 
                     problem.strip(), re.DOTALL)
    
    if match:
        name = match.group(1).strip()
        question = match.group(2).strip()
        reasoning = match.group(3).strip()
        answer = match.group(4).strip()
        
        # Convert to MathQA format
        problem_data = {
            'id': name,
            'Problem': question,
            'Rationale': reasoning,
            'correct': answer,
            'annotated_formula': '',
            'linear_formula': '',
            'category': 'probability' # Assuming all are probability problems
        }
        
        parsed_problems.append(problem_data)

# Reverse the order of problems (from hardest->easiest to easiest->hardest)
parsed_problems.reverse()

# Create DataFrame
df = pd.DataFrame(parsed_problems)

# Save as jsonl format (one JSON object per line)
with open('quant_qa_mathqa_format.jsonl', 'w') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')

# Also save as a Hugging Face dataset
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("quant_qa_dataset")

# Push to Hugging Face Hub
repo_id = "Duruo/quant_qa"  # Change this to your desired name
dataset.push_to_hub(repo_id)

# Print statistics
print(f"Total problems converted: {len(dataset)}")
print("\nSample problem in MathQA format:")
print(json.dumps(parsed_problems[0], indent=2))
print("\nData saved to quant_qa_mathqa_format.jsonl and quant_qa_dataset/")

# Save as CSV for easier viewing
df.to_csv("quant_qa_mathqa_format.csv", index=False)
print("Also saved as CSV at quant_qa_mathqa_format.csv")