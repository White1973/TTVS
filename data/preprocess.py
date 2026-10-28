import argparse
from pathlib import Path

import datasets


def make_map_fn(split, source=None):
    def process_fn(example, idx):
        data_source = example.pop("source") if source is None else source
        question = example.pop("prompt")
        solution = example.pop("answer")

        return {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}",
            },
        }

    return process_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a TTVS JSON benchmark to verl Parquet files.")
    parser.add_argument(
        "--data-source",
        required=True,
        help="Benchmark directory under data/, for example MATH-TTT or GPQA-TTT.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Training JSON filename. Defaults to train_init.json for MATH-TTT and train.json otherwise.",
    )
    parser.add_argument("--test-file", default="test.json", help="Test JSON filename.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(__file__).resolve().parent / args.data_source
    train_name = args.train_file or ("train_init.json" if args.data_source == "MATH-TTT" else "train.json")
    train_path = data_dir / train_name
    test_path = data_dir / args.test_file

    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark input file(s): {', '.join(missing)}")

    train_dataset = datasets.load_dataset("json", data_files=str(train_path), split="train")
    test_dataset = datasets.load_dataset("json", data_files=str(test_path), split="train")

    train_dataset = train_dataset.map(function=make_map_fn("train", args.data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", args.data_source), with_indices=True)

    train_output = data_dir / "train.parquet"
    test_output = data_dir / "test.parquet"
    train_dataset.to_parquet(train_output)
    test_dataset.to_parquet(test_output)
    print(f"Wrote {train_output}")
    print(f"Wrote {test_output}")


if __name__ == "__main__":
    main()
