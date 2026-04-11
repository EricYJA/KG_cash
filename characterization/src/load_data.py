from datasets import load_dataset
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_webqsp():
    print("Loading WebQSP...")
    dataset = load_dataset("rmanluo/RoG-webqsp", cache_dir=DATA_DIR)
    print(f"  Train: {len(dataset['train'])} | "
          f"Validation: {len(dataset['validation'])} | "
          f"Test: {len(dataset['test'])}")
    return dataset

def load_cwq():
    print("Loading CWQ...")
    dataset = load_dataset("rmanluo/RoG-cwq", cache_dir=DATA_DIR)
    print(f"  Train: {len(dataset['train'])} | "
          f"Validation: {len(dataset['validation'])} | "
          f"Test: {len(dataset['test'])}")
    return dataset

def inspect_sample(dataset, split='train', index=0):
    """Print a single sample so you can understand the structure."""
    sample = dataset[split][index]
    print(f"\n--- Sample {index} from {split} split ---")
    for key, value in sample.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"  {key}: {value[:3]} ... ({len(value)} items)")
        else:
            print(f"  {key}: {value}")

    # if 'graph' in sample:
    #     print("  full_graph:")
    #     for i, triple in enumerate(sample['graph']):
    #         print(f"    [{i}] {triple}")

def extract_entities(dataset, split='train'):
    """
    Returns:
      per_question: list of lists, one list of entities per question
      flat:         flat list of all entity mentions across all questions
    """
    per_question = []
    flat = []

    for sample in dataset[split]:
        entities = sample['q_entity']
        if isinstance(entities, str):
            entities = [entities]
        per_question.append(entities)
        flat.extend(entities)

    return per_question, flat
