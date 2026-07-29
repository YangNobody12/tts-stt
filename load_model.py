import argparse
import os
import sys

# Default models
DEFAULT_STT_MODEL = "unsloth/whisper-large-v3"
DEFAULT_TTS_MODEL = "unsloth/orpheus-3b-0.1-ft"
DEFAULT_SNAC_MODEL = "hubertsiuzdak/snac_24khz"

# Predefined models based on available list
AVAILABLE_MODELS = {
    "1": {
        "id": "unsloth/whisper-tiny",
        "type": "Automatic Speech Recognition (STT)",
        "params": "37.8M",
        "category": "whisper",
        "description": "Fastest Whisper model, lowest VRAM/memory footprint."
    },
    "2": {
        "id": "unsloth/whisper-base",
        "type": "Automatic Speech Recognition (STT)",
        "params": "72.6M",
        "category": "whisper",
        "description": "Lightweight Whisper model, good speed for speech recognition."
    },
    "3": {
        "id": "unsloth/whisper-small",
        "type": "Automatic Speech Recognition (STT)",
        "params": "0.2B",
        "category": "whisper",
        "description": "Balanced accuracy Whisper model for general STT tasks."
    },
    "4": {
        "id": "unsloth/whisper-large-v3-turbo",
        "type": "Automatic Speech Recognition (STT)",
        "params": "0.8B",
        "category": "whisper",
        "description": "Optimized high-accuracy Whisper model with turbo speed."
    },
    "5": {
        "id": "unsloth/whisper-large-v3",
        "type": "Automatic Speech Recognition (STT) [DEFAULT STT]",
        "params": "2B",
        "category": "whisper",
        "description": "Highest accuracy Whisper model for speech-to-text (Default STT)."
    },
    "6": {
        "id": "unsloth/orpheus-3b-0.1-ft",
        "type": "Text-to-Speech / LLM (TTS) [DEFAULT TTS]",
        "params": "3B",
        "category": "llm_tts",
        "description": "Orpheus 3B fine-tuned model for voice / speech tasks (Default TTS)."
    },
    "7": {
        "id": "hubertsiuzdak/snac_24khz",
        "type": "Neural Audio Codec (SNAC 24kHz) [DEFAULT SNAC]",
        "params": "24kHz Codec",
        "category": "snac",
        "description": "Multi-scale neural audio codec for audio tokenization & speech synthesis."
    }
}


def print_banner():
    print("=" * 65)
    print("      Model Loader CLI - Speech, Voice & Codec Models        ")
    print("=" * 65)


def list_models():
    print("\nAvailable Models:")
    print("-" * 65)
    for key, info in AVAILABLE_MODELS.items():
        print(f"  [{key}] {info['id']}")
        print(f"      Type        : {info['type']}")
        print(f"      Parameters  : {info['params']}")
        print(f"      Description : {info['description']}")
        print("-" * 65)
    print(f"  [9] All Default Models ({DEFAULT_STT_MODEL} + {DEFAULT_TTS_MODEL} + {DEFAULT_SNAC_MODEL})")
    print("-" * 65)


def resolve_model_id(user_input: str) -> list[str]:
    """Resolve selection key, alias, or direct repo name to a list of model IDs."""
    clean_input = user_input.strip().lower()
    
    # Hitting enter, typing 'default', 'defaults', 'all', 'both', or '9' returns ALL 3 default models
    if not clean_input or clean_input in ["default", "defaults", "all", "both", "9"]:
        return [DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, DEFAULT_SNAC_MODEL]
    elif clean_input in ["stt", "5"]:
        return [DEFAULT_STT_MODEL]
    elif clean_input in ["tts", "orpheus", "orpheus-3b", "6"]:
        return [DEFAULT_TTS_MODEL]
    elif clean_input in ["snac", "snac_24khz", "codec", "7"]:
        return [DEFAULT_SNAC_MODEL]
    elif clean_input in AVAILABLE_MODELS:
        return [AVAILABLE_MODELS[clean_input]["id"]]
    
    return [user_input.strip()]


def load_selected_model(model_id: str, output_dir: str = None, download_only: bool = False, device: str = "auto"):
    """Load or download specified model using standard Hugging Face transformers/huggingface_hub/snac."""
    print(f"\n[+] Selected Model ID : {model_id}")
    if output_dir:
        print(f"[+] Output Cache Directory : {os.path.abspath(output_dir)}")
    print(f"[+] Device Target          : {device}")
    print(f"[+] Download Only Mode     : {download_only}")
    print("-" * 65)

    # 1. Download Model snapshot via huggingface_hub if download_only or verification
    try:
        from huggingface_hub import snapshot_download
        print(f"[*] Downloading / verifying files from Hugging Face: {model_id}...")
        save_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir if output_dir else None,
            resume_download=True
        )
        print(f"[✓] Download completed. Saved at: {save_path}")
        if download_only:
            return save_path, None, None
    except ImportError:
        print("[!] Note: 'huggingface_hub' is not installed. Proceeding to load via 'transformers'...")
    except Exception as e:
        print(f"[!] Hugging Face Hub download note: {e}")

    if download_only:
        print("[✓] Model download finished.")
        return None, None, None

    # 2. Loading model using standard Hugging Face Transformers & SNAC libraries
    print("\n[*] Loading model into memory using standard Transformers / SNAC...")

    # Check for SNAC neural audio codec model
    if "snac" in model_id.lower():
        try:
            from snac import SNAC
            print("[*] Loading with `snac` package...")
            model = SNAC.from_pretrained(model_id).eval()
            if device != "cpu":
                import torch
                if torch.cuda.is_available():
                    model = model.to("cuda")
            print("[✓] Successfully loaded SNAC Audio Codec Model!")
            return model_id, model, None
        except ImportError:
            print("[!] Note: `snac` package not installed (`pip install snac`). Trying transformers/AutoModel...")
        except Exception as e:
            print(f"[!] SNAC library load note: {e}")

    # Standard Hugging Face transformers loading
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoTokenizer, AutoModel

        device_target = device
        if device == "auto":
            device_target = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[*] Loading via Hugging Face transformers on device '{device_target}'...")

        # Detect model category (whisper vs snac vs causal LM / TTS)
        if "whisper" in model_id.lower():
            print("[*] Loading Whisper Speech Recognition model (AutoModelForSpeechSeq2Seq)...")
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=output_dir)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                cache_dir=output_dir,
                torch_dtype=torch.float16 if device_target == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(device_target)
            print("[✓] Whisper model & processor loaded successfully!")
            return model_id, model, processor
        elif "snac" in model_id.lower():
            print("[*] Loading Neural Audio Codec (AutoModel)...")
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=output_dir,
                trust_remote_code=True
            ).to(device_target)
            print("[✓] SNAC Audio Codec Model loaded successfully!")
            return model_id, model, None
        else:
            print("[*] Loading Causal Language / Speech model (AutoModelForCausalLM)...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=output_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=output_dir,
                torch_dtype=torch.float16 if device_target == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(device_target)
            print("[✓] Model & tokenizer loaded successfully!")
            return model_id, model, tokenizer

    except ImportError:
        print("\n[!] PyTorch or Transformers is not installed in the current environment.")
        print("    To run and inference with models, please install dependencies:")
        print("    pip install torch transformers huggingface_hub snac")
    except Exception as e:
        print(f"\n[X] Failed to load model: {e}")

    return None, None, None


def interactive_menu():
    print_banner()
    list_models()
    print("  [8] Enter Custom Model Repo ID (e.g. owner/model-name)")
    print("=" * 65)

    choice = input(f"\nSelect a model number [1-9] (Default [9] All Default Models): ").strip()
    if not choice:
        print(f"[*] Default selected: All Default Models ({DEFAULT_STT_MODEL}, {DEFAULT_TTS_MODEL}, {DEFAULT_SNAC_MODEL})")
        choice = "9"

    if choice == "8":
        custom_id = input("Enter Hugging Face repository ID: ").strip()
        if not custom_id:
            print("Invalid repository ID. Exiting.")
            sys.exit(1)
        model_ids = [custom_id]
    else:
        model_ids = resolve_model_id(choice)

    dl_choice = input("Download weights only without loading into memory? (y/N): ").strip().lower()
    download_only = dl_choice == "y"

    out_dir = input("Custom output directory (leave blank for default HF cache): ").strip()
    output_dir = out_dir if out_dir else None

    for model_id in model_ids:
        load_selected_model(model_id=model_id, output_dir=output_dir, download_only=download_only)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to select, download, and load STT / TTS / SNAC Audio Codec models using standard Transformers."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="default",
        help="Model choice: index (1-9), 'default' / 'all' (all 3 default models), 'stt', 'tts', 'snac', or HF repo ID."
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available preset models and exit."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save/download model files."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download model weights to disk without loading into RAM/VRAM."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to load model onto (auto, cuda, cpu)."
    )

    args = parser.parse_args()

    if args.list:
        print_banner()
        list_models()
        return

    if len(sys.argv) == 1:
        interactive_menu()
    else:
        print_banner()
        model_ids = resolve_model_id(args.model)
        for model_id in model_ids:
            load_selected_model(
                model_id=model_id,
                output_dir=args.output_dir,
                download_only=args.download_only,
                device=args.device
            )


if __name__ == "__main__":
    main()