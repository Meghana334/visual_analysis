#!/usr/bin/env python3
"""
MiniGPT-4 Alt Text Validator - Simple Command Line Version
Auto-detects model paths, supports CPU mode
"""

import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from transformers import StoppingCriteriaList
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ===== DEFAULT PATHS =====
# Common locations where MiniGPT-4 models might be stored
DEFAULT_MODEL_PATHS = [
    # Relative to MiniGPT-4 directory
    "./llama-2-7b-chat",
    "./Llama-2-7b-chat-hf",
    "../llama-2-7b-chat-hf",
    # Common home directory locations
    str(Path.home() / "models" / "Llama-2-7b-chat-hf"),
    str(Path.home() / "models" / "llama-2-7b-chat"),
    str(Path.home() / ".cache" / "huggingface" / "hub"),
]

DEFAULT_CHECKPOINT_PATHS = [
    "./checkpoints/minigpt4_llama2_aligned.pth",
    "./checkpoints/pretrained_minigpt4_7b.pth",
    "../checkpoints/minigpt4_llama2_aligned.pth",
]

# ===== PROMPT TEMPLATES =====
VALIDATION_PROMPT = """You are a strict accessibility auditor.

FIRST: Look at the image and identify ONLY what is clearly and objectively visible.
Do NOT infer intent, emotions, symbolism, or meaning unless it is visually explicit.

SECOND: Compare the provided alt text against the image.

IMAGE ANALYSIS REQUIREMENTS:
- Identify the primary subject
- Identify secondary visible elements
- Identify any visible text (exact words)
- Identify the image type (logo, screenshot, map, product, decorative, etc.)

ALT TEXT TO EVALUATE:
"{current_alt_text}"

EVALUATION DIMENSIONS:

1. RELEVANCE (0–10)
- Does the alt text describe the SAME primary subject shown in the image?
- Does it avoid guessing intent, emotions, or meaning?
- Are all described elements visibly present?

2. ACCURACY (0–10)
- Are descriptions factually correct?
- Is visible text transcribed correctly?

3. COMPLETENESS (0–10)
- Does it include all essential visible information?
- Is the primary subject clearly identified?

4. ACCESSIBILITY QUALITY (0–10)
- Clear, objective, screen-reader friendly
- No redundant phrases like "image of"

Respond ONLY in valid JSON:

{{
  "image_type": "identified type",
  "primary_visible_subject": "what the image mainly shows",
  "relevance_score": 0,
  "accuracy_score": 0,
  "completeness_score": 0,
  "accessibility_score": 0,
  "is_adequate": false,
  "issues": ["specific problems"],
  "overall_assessment": "brief justification grounded in the image"
}}

IMPORTANT:
- is_adequate MUST be false if relevance_score < 8
- Judge ONLY against what is visible
"""

SUGGESTION_PROMPT = """You are an accessibility specialist generating compliant alt text.

STEP 1 — VISUAL GROUNDING
Describe ONLY what is clearly visible in the image:
- Primary subject
- Secondary elements
- Visible text (exact transcription)

STEP 2 — IDENTIFY IMAGE TYPE
Choose ONE:
- Logo / Brand
- Screenshot / Interface
- Map / Diagram
- Product image
- Informational image
- Decorative image

STEP 3 — WRITE ALT TEXT

RULES:
- One concise sentence
- No "image of"
- Front-load the main subject
- Accuracy > creativity

OUTPUT JSON ONLY:

{{
  "image_type": "type",
  "suggested_alt_text": "final alt text suitable for HTML",
  "why_this_is_correct": "references visible elements only"
}}
"""


def find_model_path():
    """Auto-detect Llama model path"""
    for path in DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            # Check if it's actually a model directory
            if os.path.exists(os.path.join(path, "config.json")) or \
                    os.path.exists(os.path.join(path, "pytorch_model.bin")) or \
                    any(f.endswith(".safetensors") for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
                return os.path.abspath(path)
    return None


def find_checkpoint_path():
    """Auto-detect MiniGPT-4 checkpoint path"""
    for path in DEFAULT_CHECKPOINT_PATHS:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def setup_seeds(seed=42):
    """Setup random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def initialize_minigpt4_manual(
        llama_model_path,
        checkpoint_path,
        model_type='llama2',
        use_cpu=False,
        gpu_id=0,
        low_resource=False
):
    """
    Initialize MiniGPT-4 model without config file
    """
    print("\n" + "=" * 60)
    print("INITIALIZING MINIGPT-4 MODEL")
    print("=" * 60)
    print(f"Llama model: {llama_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {model_type}")
    print(f"Device: {'CPU' if use_cpu else f'GPU {gpu_id}'}")
    print(f"Low resource mode: {low_resource}")
    print("This may take 30-60 seconds...\n")

    # Setup seeds
    setup_seeds(42)

    # Select conversation template
    conv_dict = {
        'vicuna': CONV_VISION_Vicuna0,
        'llama2': CONV_VISION_LLama2
    }
    CONV_VISION = conv_dict[model_type]

    # Create model config manually
    from omegaconf import OmegaConf

    device = 'cpu' if use_cpu else f'cuda:{gpu_id}'

    model_config = OmegaConf.create({
        'arch': 'mini_gpt4',
        'model_type': f'pretrain_{model_type}',
        'freeze_vit': True,
        'freeze_qformer': True,
        'max_txt_len': 160,
        'end_sym': '###',
        'low_resource': low_resource or use_cpu,
        'device_8bit': gpu_id if not use_cpu else 0,
        'prompt_template': '###Human: {} ###Assistant: ',
        'ckpt': checkpoint_path,
        'llama_model': llama_model_path,
    })

    # Initialize model
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)

    # Initialize visual processor
    from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor
    vis_processor = Blip2ImageTrainProcessor(image_size=224)

    # Setup stopping criteria
    stop_words_ids = [[835], [2277, 29937]]
    if use_cpu:
        stop_words_ids = [torch.tensor(ids).to(device='cpu') for ids in stop_words_ids]
    else:
        stop_words_ids = [torch.tensor(ids).to(device=f'cuda:{gpu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # Create chat instance
    chat = Chat(
        model,
        vis_processor,
        device=device,
        stopping_criteria=stopping_criteria
    )

    print("✓ MiniGPT-4 model loaded successfully!")
    print(f"  Model type: {model_type}")
    print(f"  Architecture: mini_gpt4")
    print(f"  Device: {device}\n")

    return chat, CONV_VISION


def validate_alt_text(chat, conv_template, image_path, current_alt_text, num_beams=1, temperature=0.3):
    """Validate existing alt text against the image using MiniGPT-4"""

    # Create new conversation state
    chat_state = conv_template.copy()
    img_list = []

    # Load and upload image
    image = Image.open(image_path).convert('RGB')
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)

    # Prepare validation prompt
    prompt = VALIDATION_PROMPT.format(current_alt_text=current_alt_text)

    # Ask the question
    chat.ask(prompt, chat_state)

    # Get answer
    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=500,
        max_length=2000
    )[0]

    # Extract JSON from response
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            validation = json.loads(json_str)
        else:
            validation = {
                "is_adequate": False,
                "relevance_score": 0,
                "accuracy_score": 0,
                "completeness_score": 0,
                "accessibility_score": 0,
                "issues": ["Failed to parse validation response"],
                "overall_assessment": response
            }
    except json.JSONDecodeError:
        validation = {
            "is_adequate": False,
            "relevance_score": 0,
            "accuracy_score": 0,
            "completeness_score": 0,
            "accessibility_score": 0,
            "issues": ["Failed to parse validation response"],
            "overall_assessment": response
        }

    return validation


def generate_alt_text_suggestion(chat, conv_template, image_path, num_beams=1, temperature=0.5):
    """Generate improved alt text suggestion using MiniGPT-4"""

    # Create new conversation state
    chat_state = conv_template.copy()
    img_list = []

    # Load and upload image
    image = Image.open(image_path).convert('RGB')
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)

    # Ask for suggestion
    chat.ask(SUGGESTION_PROMPT, chat_state)

    # Get answer
    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=600,
        max_length=2000
    )[0]

    # Extract JSON from response
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            suggestion = json.loads(json_str)
        else:
            suggestion = {
                "image_type": "unknown",
                "suggested_alt_text": response,
                "detailed_description": "",
                "why_this_is_correct": "Raw response (JSON parsing failed)"
            }
    except json.JSONDecodeError:
        suggestion = {
            "image_type": "unknown",
            "suggested_alt_text": response,
            "detailed_description": "",
            "why_this_is_correct": "Raw response (JSON parsing failed)"
        }

    return suggestion


def process_csv(chat, conv_template, csv_file, output_file, num_beams=1, temperature=0.7):
    """Process CSV file with images and alt text"""
    print("\n" + "=" * 60)
    print("PROCESSING CSV FILE")
    print("=" * 60)

    if not os.path.exists(csv_file):
        print(f"✗ Error: CSV file '{csv_file}' not found!")
        return

    results = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)

        print(f"Found {total} images to process\n")

        # Check for required columns
        field_names = reader.fieldnames

        path_col = 'image_path'
        alt_col = 'current_alt_text'

        if 'Image Path' in field_names and 'Alt Text' in field_names:
            path_col = 'Image Path'
            alt_col = 'Alt Text'
            print("Detected standard report format (Image Path, Alt Text)")
        elif 'image_path' in field_names and 'current_alt_text' in field_names:
            print("Detected simple format (image_path, current_alt_text)")
        else:
            print("Warning: Could not detect standard columns. Falling back to 'image_path' and 'current_alt_text'")

        for idx, row in enumerate(rows, 1):
            image_path = row.get(path_col, '').strip()
            current_alt = row.get(alt_col, '').strip()

            print(f"\n[{idx}/{total}] Processing: {image_path}")
            print(f"Current alt text: {current_alt[:80]}{'...' if len(current_alt) > 80 else ''}")

            if not os.path.exists(image_path):
                print(f"  ✗ Image not found, skipping...")
                results.append({
                    'image_path': image_path,
                    'current_alt_text': current_alt,
                    'status': 'ERROR',
                    'error': 'Image file not found',
                    'is_adequate': 'N/A',
                    'relevance_score': 'N/A',
                    'accuracy_score': 'N/A',
                    'completeness_score': 'N/A',
                    'accessibility_score': 'N/A',
                    'issues': 'Image file not found',
                    'suggested_alt_text': '',
                    'image_type': '',
                    'reasoning': ''
                })
                continue

            try:
                # Step 1: Validate current alt text
                print("  → Validating current alt text...")
                validation = validate_alt_text(chat, conv_template, image_path, current_alt, num_beams, temperature=0.3)

                is_adequate = validation.get('is_adequate', False)
                relevance = validation.get('relevance_score', 0)
                accuracy = validation.get('accuracy_score', 0)
                completeness = validation.get('completeness_score', 0)
                accessibility = validation.get('accessibility_score', 0)

                print(
                    f"  → Scores: Relevance={relevance}/10, Accuracy={accuracy}/10, Completeness={completeness}/10, Accessibility={accessibility}/10")
                print(f"  → Adequate: {'YES ✓' if is_adequate else 'NO ✗'}")

                # Step 2: Generate suggestion if needed
                suggestion = None
                if not is_adequate:
                    print("  → Generating improved alt text suggestion...")
                    suggestion = generate_alt_text_suggestion(chat, conv_template, image_path, num_beams, temperature)
                    print(f"  → Suggestion: {suggestion.get('suggested_alt_text', '')[:80]}...")

                # Store results
                result_row = {
                    'image_path': image_path,
                    'current_alt_text': current_alt,
                    'status': 'ADEQUATE' if is_adequate else 'NEEDS_IMPROVEMENT',
                    'is_adequate': is_adequate,
                    'relevance_score': relevance,
                    'accuracy_score': accuracy,
                    'completeness_score': completeness,
                    'accessibility_score': accessibility,
                    'issues': '; '.join(validation.get('issues', [])),
                    'overall_assessment': validation.get('overall_assessment', ''),
                    'suggested_alt_text': suggestion.get('suggested_alt_text', '') if suggestion else '',
                    'detailed_description': suggestion.get('detailed_description', '') if suggestion else '',
                    'image_type': suggestion.get('image_type', '') if suggestion else '',
                    'reasoning': suggestion.get('why_this_is_correct', '') if suggestion else ''
                }

                results.append(result_row)

            except Exception as e:
                print(f"  ✗ Error processing image: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'image_path': image_path,
                    'current_alt_text': current_alt,
                    'status': 'ERROR',
                    'error': str(e),
                    'is_adequate': 'N/A',
                    'relevance_score': 'N/A',
                    'accuracy_score': 'N/A',
                    'completeness_score': 'N/A',
                    'accessibility_score': 'N/A',
                    'issues': str(e),
                    'suggested_alt_text': '',
                    'image_type': '',
                    'reasoning': ''
                })

    # Write results to output CSV
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    output_fields = [
        'image_path',
        'current_alt_text',
        'status',
        'is_adequate',
        'relevance_score',
        'accuracy_score',
        'completeness_score',
        'accessibility_score',
        'issues',
        'overall_assessment',
        'suggested_alt_text',
        'detailed_description',
        'image_type',
        'reasoning'
    ]

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Results saved to: {output_file}")

    # Print summary
    adequate_count = sum(1 for r in results if r['is_adequate'] is True)
    needs_improvement = sum(1 for r in results if r['status'] == 'NEEDS_IMPROVEMENT')
    error_count = sum(1 for r in results if r.get('status') == 'ERROR')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(results)}")
    print(f"✓ Adequate alt text: {adequate_count}")
    print(f"✗ Needs improvement: {needs_improvement}")
    if error_count > 0:
        print(f"⚠ Errors: {error_count}")
    print(f"\nAverage scores:")

    valid_results = [r for r in results if r['status'] != 'ERROR']
    if valid_results:
        num_valid = len([r for r in valid_results if isinstance(r['relevance_score'], (int, float))])
        if num_valid > 0:
            avg_relevance = sum(r['relevance_score'] for r in valid_results if
                                isinstance(r['relevance_score'], (int, float))) / num_valid
            avg_accuracy = sum(
                r['accuracy_score'] for r in valid_results if isinstance(r['accuracy_score'], (int, float))) / num_valid
            avg_completeness = sum(r['completeness_score'] for r in valid_results if
                                   isinstance(r['completeness_score'], (int, float))) / num_valid
            avg_accessibility = sum(r['accessibility_score'] for r in valid_results if
                                    isinstance(r['accessibility_score'], (int, float))) / num_valid

            print(f"  Relevance: {avg_relevance:.1f}/10")
            print(f"  Accuracy: {avg_accuracy:.1f}/10")
            print(f"  Completeness: {avg_completeness:.1f}/10")
            print(f"  Accessibility: {avg_accessibility:.1f}/10")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT-4 Alt Text Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage (auto-detect models):
  python3 alttext_validator.py --input images.csv --output-dir results

  # With CPU mode:
  python3 alttext_validator.py --input images.csv --output-dir results --cpu

  # Specify model paths manually:
  python3 alttext_validator.py --input images.csv \\
      --llama-model /path/to/Llama-2-7b-chat-hf \\
      --checkpoint /path/to/minigpt4.pth
        """
    )

    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output-dir", help="Directory to save the output CSV (default: same as input)")
    parser.add_argument("--llama-model", help="Path to Llama model (auto-detected if not specified)")
    parser.add_argument("--checkpoint", help="Path to MiniGPT-4 checkpoint (auto-detected if not specified)")
    parser.add_argument("--model-type", choices=['llama2', 'vicuna'], default='llama2',
                        help="Model type (default: llama2)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (slower but works without GPU)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (default: 0.7)")
    parser.add_argument("--low-resource", action="store_true", help="Use 8-bit quantization for low memory")

    args = parser.parse_args()

    input_csv = args.input

    # Auto-detect model paths if not provided
    llama_model_path = args.llama_model
    checkpoint_path = args.checkpoint

    if not llama_model_path:
        print("Attempting to auto-detect Llama model path...")
        llama_model_path = find_model_path()
        if llama_model_path:
            print(f"✓ Found Llama model at: {llama_model_path}")
        else:
            print("✗ Could not auto-detect Llama model path!")
            print("Please specify --llama-model /path/to/Llama-2-7b-chat-hf")
            return

    if not checkpoint_path:
        print("Attempting to auto-detect MiniGPT-4 checkpoint...")
        checkpoint_path = find_checkpoint_path()
        if checkpoint_path:
            print(f"✓ Found checkpoint at: {checkpoint_path}")
        else:
            print("✗ Could not auto-detect checkpoint path!")
            print("Please specify --checkpoint /path/to/minigpt4_checkpoint.pth")
            return

    # Determine output file
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(input_csv)) or '.'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"alttext_validation_{timestamp}.csv"
    output_csv = os.path.join(output_dir, output_filename)

    print("\n" + "=" * 60)
    print("MiniGPT-4 Alt Text Validator")
    print("=" * 60)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Llama model: {llama_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {'CPU' if args.cpu else f'GPU {args.gpu_id}'}")
    print(f"Beam search: {args.num_beams}")
    print(f"Temperature: {args.temperature}")
    print(f"Low resource: {args.low_resource or args.cpu}")
    print("=" * 60)

    # Initialize MiniGPT-4
    chat, conv_template = initialize_minigpt4_manual(
        llama_model_path=llama_model_path,
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        use_cpu=args.cpu,
        gpu_id=args.gpu_id,
        low_resource=args.low_resource
    )

    # Process CSV
    process_csv(chat, conv_template, input_csv, output_csv, args.num_beams, args.temperature)

    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()