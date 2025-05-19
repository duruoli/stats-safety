import json
from pathlib import Path

# Paths to data files
q_path = "forecastbench-datasets/datasets/question_sets/2024-12-08-llm.json"
r_path = "forecastbench-datasets/datasets/resolution_sets/2024-12-08_resolution_set.json"


# Load question set
with open(q_path) as f:
    llm_data = json.load(f)

# Load resolution set
with open(r_path) as f:
    resolution_data = json.load(f)


# Build lookup of resolved answers for single-ID questions
resolution_lookup = {}
for res in resolution_data['resolutions']:
    # Only single-ID (string) entries
    if isinstance(res['id'], str):
        resolution_lookup[res['id']] = res['resolved_to']


# Filter to the 500 single-ID questions
single_questions = [q for q in llm_data['questions'] if isinstance(q['id'], str)]

# Assemble HuggingFace-format records
records = []
for q in single_questions:
    qid = q['id']
    answer = resolution_lookup.get(qid)
    if answer is None:
        continue  # skip if no resolution available

    # Collect all relevant fields, including the source_intro for use as a prompt prefix
    records.append({
        'id': qid,
        'question': q.get('question') or q.get('question_text'),  # choose available field
        'background': q.get('background'),
        'source': q.get('source'),
        'source_intro': q.get('source_intro'),  # include this as part of the prompt
        'answer': int(answer),
    })

# Write out as JSONL
out_path = Path("forecastbench_single_questions_2024-12-08.jsonl")
with out_path.open('w') as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Saved {len(records)} records to {out_path}")
