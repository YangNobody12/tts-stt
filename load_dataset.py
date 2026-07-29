import argparse
import os
import sys

DEFAULT_DATASET = "Thanarit/Thai-Voice-Test7"

PRESET_DATASETS = {
    "1": {
        "id": "Thanarit/Thai-Voice-Test7",
        "name": "Thai Voice Test7 (DEFAULT)",
        "description": "Thai speech audio dataset for Speech-to-Text & Voice training."
    },
    "2": {
        "id": "google/fleurs",
        "name": "FLEURS Thai (th_th)",
        "description": "Google FLEURS speech dataset for Thai speech recognition."
    },
    "3": {
        "id": "mozilla-foundation/common_voice_11_0",
        "name": "Mozilla Common Voice Thai",
        "description": "Mozilla Common Voice open Thai speech dataset."
    }
}


def print_banner():
    print("=" * 65)
    print("      Dataset & Metrics Loader CLI - Audio & Evaluation       ")
    print("=" * 65)


def list_datasets():
    print("\nPreset Datasets:")
    print("-" * 65)
    for key, info in PRESET_DATASETS.items():
        print(f"  [{key}] {info['id']}")
        print(f"      Name        : {info['name']}")
        print(f"      Description : {info['description']}")
        print("-" * 65)


def resolve_dataset_id(user_input: str) -> str:
    """Resolve key (1-3) or direct dataset ID string."""
    clean_input = user_input.strip()
    if not clean_input or clean_input.lower() in ["default", "1"]:
        return DEFAULT_DATASET
    if clean_input in PRESET_DATASETS:
        return PRESET_DATASETS[clean_input]["id"]
    return clean_input


def predownload_metrics():
    """Download and cache evaluation metrics (wer, cer) for offline compute nodes."""
    print("\n[*] Pre-downloading evaluation metrics (wer, cer)...")
    try:
        import evaluate
        metric_wer = evaluate.load("wer")
        print("[✓] Successfully downloaded & cached 'wer' metric!")
        metric_cer = evaluate.load("cer")
        print("[✓] Successfully downloaded & cached 'cer' metric!")
        return metric_wer, metric_cer
    except ImportError:
        print("[!] Note: 'evaluate' package not installed. Skipping metrics cache (`pip install evaluate jiwer`).")
    except Exception as e:
        print(f"[!] Evaluation metric cache note: {e}")
    return None, None


def load_and_inspect_dataset(dataset_id: str, split: str = None, save_dir: str = None, download_only: bool = False):
    """Load or download Hugging Face dataset and inspect its structure."""
    print(f"\n[+] Selected Dataset ID : {dataset_id}")
    print(f"[+] Requested Split     : {split if split else 'All available'}")
    if save_dir:
        print(f"[+] Output Directory   : {os.path.abspath(save_dir)}")
    print(f"[+] Download Only Mode : {download_only}")
    print("-" * 65)

    # Pre-download metrics (wer, cer) so evaluate.load("wer") works offline on compute nodes
    predownload_metrics()

    # 1. Download snapshot files directly via huggingface_hub if download_only
    if download_only:
        try:
            from huggingface_hub import snapshot_download
            print(f"\n[*] Downloading raw dataset files for: {dataset_id}...")
            save_path = snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=save_dir if save_dir else None
            )
            print(f"[✓] Raw dataset files downloaded successfully to: {save_path}")
            return save_path
        except Exception as e:
            print(f"[!] Hugging Face Hub download failed/note: {e}")

    # 2. Load dataset via `datasets` library
    try:
        from datasets import load_dataset
        print(f"\n[*] Loading dataset via Hugging Face `datasets`...")
        
        # Load dataset
        if split:
            dataset = load_dataset(dataset_id, split=split)
        else:
            dataset = load_dataset(dataset_id)

        print("[✓] Dataset loaded successfully!\n")
        print("=" * 65)
        print("                      DATASET INFORMATION                     ")
        print("=" * 65)

        if hasattr(dataset, "keys"):
            print(f"Available Splits : {list(dataset.keys())}")
            for s_name in dataset.keys():
                ds_split = dataset[s_name]
                print(f"\n--- Split: '{s_name}' ---")
                print(f"Total Rows : {len(ds_split)}")
                print(f"Columns    : {ds_split.column_names}")
                if len(ds_split) > 0:
                    print(f"Sample Entry (Row 0) keys: {list(ds_split[0].keys())}")
                    # Print transcript sample if present
                    for text_col in ["sentence", "text", "transcript", "normalized_text"]:
                        if text_col in ds_split[0]:
                            print(f"Sample Transcript ({text_col}): \"{ds_split[0][text_col]}\"")
                            break
        else:
            print(f"Total Rows : {len(dataset)}")
            print(f"Columns    : {dataset.column_names}")
            if len(dataset) > 0:
                print(f"Sample Entry (Row 0) keys: {list(dataset[0].keys())}")
                for text_col in ["sentence", "text", "transcript", "normalized_text"]:
                    if text_col in dataset[0]:
                        print(f"Sample Transcript ({text_col}): \"{dataset[0][text_col]}\"")
                        break

        # Save dataset to local disk if save_dir is specified
        if save_dir:
            print(f"\n[*] Saving processed dataset to: {os.path.abspath(save_dir)}...")
            dataset.save_to_disk(save_dir)
            print(f"[✓] Dataset saved to disk successfully!")

        return dataset

    except ImportError:
        print("\n[!] The `datasets` library is not installed in the current environment.")
        print("    Please install it using pip:")
        print("    pip install datasets soundfile librosa evaluate jiwer")
    except Exception as e:
        print(f"\n[X] Failed to load dataset: {e}")

    return None


def interactive_menu():
    print_banner()
    list_datasets()
    print("  [4] Enter Custom Dataset Repo ID (e.g. owner/dataset-name)")
    print("=" * 65)

    choice = input(f"\nSelect a dataset number [1-4] (Default [1] {DEFAULT_DATASET}): ").strip()
    if not choice:
        print(f"[*] Default selected: {DEFAULT_DATASET}")
        dataset_id = DEFAULT_DATASET
    elif choice == "4":
        custom_id = input("Enter Hugging Face Dataset repository ID: ").strip()
        if not custom_id:
            print("Invalid repository ID. Exiting.")
            sys.exit(1)
        dataset_id = custom_id
    else:
        dataset_id = resolve_dataset_id(choice)

    split = input("Enter split to load (e.g. train, test, validation, or leave blank for all): ").strip()
    split_val = split if split else None

    dl_choice = input("Download raw files only without loading into memory? (y/N): ").strip().lower()
    download_only = dl_choice == "y"

    out_dir = input("Custom save directory (leave blank to skip local saving): ").strip()
    save_dir = out_dir if out_dir else None

    load_and_inspect_dataset(dataset_id=dataset_id, split=split_val, save_dir=save_dir, download_only=download_only)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to select, download, and inspect speech datasets & evaluation metrics (Default: Thanarit/Thai-Voice-Test7 + WER/CER)."
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset choice by index (1-3) or Hugging Face repo ID (default: '{DEFAULT_DATASET}')."
    )
    parser.add_argument(
        "-s", "--split",
        type=str,
        default=None,
        help="Dataset split to load (e.g. 'train', 'test', 'validation')."
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available preset datasets and exit."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save downloaded raw files or processed dataset."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download raw dataset files and evaluation metrics from Hugging Face without loading into RAM."
    )

    args = parser.parse_args()

    if args.list:
        print_banner()
        list_datasets()
        return

    if len(sys.argv) == 1:
        interactive_menu()
    else:
        print_banner()
        dataset_id = resolve_dataset_id(args.dataset)
        load_and_inspect_dataset(
            dataset_id=dataset_id,
            split=args.split,
            save_dir=args.output_dir,
            download_only=args.download_only
        )


if __name__ == "__main__":
    main()
