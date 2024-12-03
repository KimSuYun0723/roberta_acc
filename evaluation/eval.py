import json
import os
import torch
from transformers import RobertaTokenizer, RobertaForMultipleChoice, AutoConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import jsonlines
import wandb 

# Custom Dataset for BBQ
class BBQDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=512, num_choices=3):
        with jsonlines.open(file_path, "r") as reader:  # JSONL 파일 읽기
            self.data = [line for line in reader]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_choices = num_choices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        inputs = self.tokenizer(
            [example["context"] + " " + example["query"]] * self.num_choices,
            [example[f"option_{i}"] for i in range(self.num_choices)],   
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        print("Input IDs shape:", inputs["input_ids"].shape)  # (num_choices, seq_len)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(example["label"]),
            "example_id": idx
        }

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])  # (batch_size, num_choices, seq_len)
    attention_mask = torch.stack([item["attention_mask"] for item in batch])  # (batch_size, num_choices, seq_len)
    labels = torch.tensor([item["labels"] for item in batch])  # (batch_size,)
    example_ids = [item["example_id"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_id": example_ids,
    }        

# Evaluation Function
def evaluate(model, dataloader, output_dir, device="cuda"):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)  # (batch_size, num_choices, seq_len)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 모델 출력
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (batch_size, num_choices)
            predictions = torch.argmax(logits, dim=-1)  # (batch_size,)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                results.append({
                    "example_id": batch["example_id"][i],
                    "label": labels[i].item(),
                    "prediction": predictions[i].item(),
                    "logits": logits[i].tolist(),
                })

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=4)
    return accuracy

# Main Function
def main(args):
    wandb.init(
        project="BBQ Evaluation",
        name=f"Evaluation-{args.output_dir.split('/')[-1]}",
        config={
            "model_name": args.model_name_or_path,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "num_choices": args.num_choices,
        }
    )
    print(f"Loading model from {args.model_name_or_path}")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")

    # Config
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        revision=args.model_revision,
    )

    tokenizer = RobertaTokenizer.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
    )

    #if bert
    """model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        config=config,
        revision=args.model_revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )"""
    #if unlog
    model = RobertaForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        config=config,
        revision=args.model_revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    dataset = BBQDataset(
        file_path=args.data_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        num_choices=args.num_choices,
    )


    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn
        )
    evaluate(model, dataloader, output_dir=args.output_dir, device=args.device)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the BBQ validation JSONL file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_choices", type=int, default=3, help="Number of answer choices")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path")
    parser.add_argument("--model_revision", type=str, default="main", help="Model version to use")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer")
    args = parser.parse_args()

    try: 
        main(args)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        raise