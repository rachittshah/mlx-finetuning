# Data Format

## Training Data Format

Training data should be in JSONL format (one JSON object per line) with conversational structure:

```json
{"messages": [
  {"role": "user", "content": "Your question or prompt here"},
  {"role": "assistant", "content": "Expected model response here"}
]}
```

## Example: Classification Task

```json
{"messages": [{"role": "user", "content": "Classify this sentiment: The product exceeded my expectations!"}, {"role": "assistant", "content": "Positive"}]}
{"messages": [{"role": "user", "content": "Classify this sentiment: Terrible experience, would not recommend."}, {"role": "assistant", "content": "Negative"}]}
```

## Example: Question Answering

```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "Explain photosynthesis briefly."}, {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."}]}
```

## Example: Judge Model

```json
{"messages": [{"role": "user", "content": "Compare these two answers:\n\nQuestion: What is 2+2?\nAnswer A: 4\nAnswer B: 5\n\nWhich is better?"}, {"role": "assistant", "content": "8 2\n\nAnswer A is correct (2+2=4), while Answer B is incorrect. Answer A demonstrates accurate mathematical knowledge.\n\n**Verdict:** A"}]}
```

## Data Split

Organize your data into training and validation sets:

- `train.jsonl`: Training examples (80-90% of data)
- `valid.jsonl`: Validation examples (10-20% of data)

## Data Quality Tips

1. **Consistency**: Maintain consistent format across all examples
2. **Quality over Quantity**: 1000 high-quality examples > 10000 noisy examples
3. **Diversity**: Include varied examples covering different aspects of your task
4. **Balance**: For classification, ensure balanced representation of classes
5. **Length**: Keep examples within your sequence length limit (typically 512-2048 tokens)

## Preparing Your Data

```python
import json

# Example: Convert CSV to JSONL
import pandas as df

df = pd.read_csv('your_data.csv')
with open('train.jsonl', 'w') as f:
    for _, row in df.iterrows():
        example = {
            "messages": [
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['completion']}
            ]
        }
        f.write(json.dumps(example) + '\n')
```

## Data Validation

Before training, validate your data:

```bash
# Check format
python -c "import json; [json.loads(line) for line in open('train.jsonl')]"

# Count examples
wc -l train.jsonl valid.jsonl

# Check average length
python -c "import json; lens=[len(json.loads(l)['messages'][0]['content']) for l in open('train.jsonl')]; print(f'Avg: {sum(lens)/len(lens):.0f}, Max: {max(lens)}')"
```
