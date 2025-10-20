#!/usr/bin/env python3
"""
Upload the LoRA embedding model to HuggingFace Hub

Usage:
    export HF_TOKEN=hf_your_token_here
    python scripts/upload_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "embedding_lora" / "walmart-amazon"

def upload_model():
    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("* ERROR: HF_TOKEN environment variable not set!")
        print("\nPlease run:")
        print("  export HF_TOKEN=hf_your_token_here")
        print("  python scripts/upload_to_hf.py")
        return

    print("=" * 80)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("=" * 80)

    # Get username
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]

    repo_id = f"{username}/walmart-amazon-product-matcher-lora"

    print(f"\n* Repository: {repo_id}")
    print(f"* Model path: {MODEL_PATH}")

    # Create repo
    print(f"\n🔨 Creating repository...")
    try:
        create_repo(repo_id, token=token, exist_ok=True)
        print(f"* Repository created/exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠*  Repository may already exist: {e}")

    # Upload model
    print(f"\n⬆*  Uploading model files...")
    api.upload_folder(
        folder_path=str(MODEL_PATH),
        repo_id=repo_id,
        token=token,
        commit_message="Upload LoRA fine-tuned embedding model for Walmart-Amazon product matching"
    )

    print(f"\n* UPLOAD COMPLETE!")
    print(f"\n* Your model is now available at:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n* To use it:")
    print(f"   from sentence_transformers import SentenceTransformer")
    print(f"   model = SentenceTransformer('{repo_id}')")

    # Create model card
    model_card = f"""---
tags:
- sentence-transformers
- product-matching
- walmart
- amazon
- lora
- embedding
library_name: sentence-transformers
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# Walmart-Amazon Product Matcher (LoRA Fine-tuned)

This is a LoRA fine-tuned embedding model for product matching between Walmart and Amazon catalogs.

## Model Details

- **Base Model:** sentence-transformers/all-MiniLM-L6-v2
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Trainable Parameters:** 337,920 (1.47% of 23M total)
- **Task:** Product matching / semantic similarity
- **Training Time:** 19 seconds on Apple M4 Pro

## Use Cases

- Cross-retailer product matching
- Product deduplication
- Price comparison tools
- Inventory management

## Usage

```python
from sentence_transformers import SentenceTransformer
import torch

# Load model
model = SentenceTransformer('{repo_id}')

# Encode products
walmart_emb = model.encode("Samsung 55-Inch 4K TV UN55CU7000")
amazon_emb = model.encode("Samsung 55 Inch Crystal 4K UHD TV UN55CU7000")

# Compute similarity
similarity = torch.nn.functional.cosine_similarity(
    torch.tensor(walmart_emb).unsqueeze(0),
    torch.tensor(amazon_emb).unsqueeze(0)
).item()

print(f"Similarity: {{similarity:.3f}}")  # Output: 0.918
```

## Performance

Tested on synthetic Walmart-Amazon product pairs:

| Test Case | Match Similarity | No-Match Similarity | Result |
|-----------|-----------------|-------------------|--------|
| Samsung TV | 0.918 | 0.639 | * Correct |
| AirPods Pro | 0.973 | 0.318 | * Correct |

## Training Details

- **Training Data:** 12 synthetic Walmart-Amazon product pairs
- **Validation Data:** 4 pairs
- **Loss Function:** Triplet Loss
- **Epochs:** 10
- **Batch Size:** 8

## Citation

```bibtex
@misc{{walmart-amazon-lora-2025,
  title={{Walmart-Amazon Product Matcher (LoRA Fine-tuned)}},
  author={{{username}}},
  year={{2025}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

Same as base model: Apache 2.0
"""

    # Upload model card
    print(f"\n* Creating model card...")
    with open(MODEL_PATH / "README.md", "w") as f:
        f.write(model_card)

    api.upload_file(
        path_or_fileobj=str(MODEL_PATH / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message="Add model card"
    )

    print(f"* Model card uploaded!")
    print(f"\n* ALL DONE! Check out your model: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_model()
