#!/usr/bin/env python3
"""
NuMarkdown-8B Alt Text Validator
- Caches models locally (no re-download)
- Loads images and alt text from CSV
- Validates alt text quality
- Generates suggestions for improvements
"""

import os
import csv
import json
import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from datetime import datetime

# ===== CONFIGURATION =====
MODEL_CACHE_DIR = Path.home() / ".cache" / "numarkdown_models"
MODEL_CACHE_DIR = Path.home() / ".cache" / "numarkdown_models"

# GPU Settings - Adjust based on your hardware
GPU_LAYERS = 35  # 0 for CPU-only, 35 for strong GPU, -1 for all layers

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
- No redundant phrases like “image of”

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
- No “image of”
- Front-load the main subject
- Accuracy > creativity

OUTPUT JSON ONLY:

{{
  "image_type": "type",
  "suggested_alt_text": "final alt text suitable for HTML",
  "why_this_is_correct": "references visible elements only"
}}
"""


def download_models():
    """Download and cache models (only runs once)"""
    print("\n" + "=" * 60)
    print("MODEL SETUP")
    print("=" * 60)

    # Create cache directory if it doesn't exist
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if models are already cached
    text_model_cache = MODEL_CACHE_DIR / "NuMarkdown-8B-Thinking-Q4_K_M.gguf"
    image_encoder_cache = MODEL_CACHE_DIR / "mmproj-BF16.gguf"

    if text_model_cache.exists() and image_encoder_cache.exists():
        print("✓ Models found in cache!")
        print(f"  Text model: {text_model_cache}")
        print(f"  Image encoder: {image_encoder_cache}")
        return str(text_model_cache), str(image_encoder_cache)

    # Download models if not cached
    print("Downloading models (this happens only once)...")
    print("This may take several minutes for ~5GB of data...\n")

    print("1/2 Downloading text model...")
    text_model = hf_hub_download(
        repo_id="numind/NuMarkdown-8B-Thinking-GGUF",
        filename="NuMarkdown-8B-Thinking-Q4_K_M.gguf",
        cache_dir=str(MODEL_CACHE_DIR)
    )
    print(f"✓ Text model downloaded: {text_model}")

    print("\n2/2 Downloading image encoder...")
    image_encoder = hf_hub_download(
        repo_id="numind/NuMarkdown-8B-Thinking-GGUF",
        filename="mmproj-BF16.gguf",
        cache_dir=str(MODEL_CACHE_DIR)
    )
    print(f"✓ Image encoder downloaded: {image_encoder}")

    print(f"\n✓ Models cached to: {MODEL_CACHE_DIR}")
    print("  (Future runs will use cached models)\n")

    return text_model, image_encoder


def initialize_model(text_model_path, image_encoder_path):
    """Initialize the LLM with cached models"""
    print("\n" + "=" * 60)
    print("LOADING MODEL INTO MEMORY")
    print("=" * 60)
    print("This may take 30-60 seconds...\n")

    llm = Llama(
        model_path=text_model_path,
        mmproj_path=image_encoder_path,
        n_ctx=4096,
        n_gpu_layers=GPU_LAYERS,
        verbose=False,
    )

    print("✓ Model loaded successfully!")
    print(f"  GPU layers: {GPU_LAYERS}")
    print(f"  Context window: 4096 tokens\n")

    return llm


def validate_alt_text(llm, image_path, current_alt_text):
    """Validate existing alt text against the image"""
    prompt = VALIDATION_PROMPT.format(current_alt_text=current_alt_text)

    result = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{os.path.abspath(image_path)}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=500,
        temperature=0.3,  # Lower temperature for more consistent validation
    )

    response = result["choices"][0]["message"]["content"]

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            validation = json.loads(json_str)
        else:
            # Fallback if no JSON found
            validation = {
                "is_adequate": False,
                "accuracy_score": 0,
                "completeness_score": 0,
                "accessibility_score": 0,
                "issues": ["Failed to parse validation response"],
                "overall_assessment": response
            }
    except json.JSONDecodeError:
        validation = {
            "is_adequate": False,
            "accuracy_score": 0,
            "completeness_score": 0,
            "accessibility_score": 0,
            "issues": ["Failed to parse validation response"],
            "overall_assessment": response
        }

    return validation


def generate_alt_text_suggestion(llm, image_path):
    """Generate improved alt text suggestion"""
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{os.path.abspath(image_path)}"},
                    },
                    {
                        "type": "text",
                        "text": SUGGESTION_PROMPT,
                    },
                ],
            }
        ],
        max_tokens=600,
        temperature=0.5,
    )

    response = result["choices"][0]["message"]["content"]

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
                "reasoning": "Raw response (JSON parsing failed)"
            }
    except json.JSONDecodeError:
        suggestion = {
            "image_type": "unknown",
            "suggested_alt_text": response,
            "detailed_description": "",
            "reasoning": "Raw response (JSON parsing failed)"
        }

    return suggestion


def process_csv(llm, csv_file, output_file):
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
        # Map input columns to expected variables
        # Default expectation: 'image_path', 'current_alt_text'
        # Actual input might be: 'Image Path', 'Alt Text'
        
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
                    'accuracy_score': 'N/A',
                    'completeness_score': 'N/A',
                    'accessibility_score': 'N/A',
                    'issues': 'Image file not found',
                    'suggested_alt_text': '',
                    'image_type': '',
                    'reasoning': ''
                })
                continue

            # Step 1: Validate current alt text
            print("  → Validating current alt text...")
            validation = validate_alt_text(llm, image_path, current_alt)

            is_adequate = validation.get('is_adequate', False)
            accuracy = validation.get('accuracy_score', 0)
            completeness = validation.get('completeness_score', 0)
            accessibility = validation.get('accessibility_score', 0)

            print(
                f"  → Scores: Accuracy={accuracy}/10, Completeness={completeness}/10, Accessibility={accessibility}/10")
            print(f"  → Adequate: {'YES ✓' if is_adequate else 'NO ✗'}")

            # Step 2: Generate suggestion if needed
            suggestion = None
            if not is_adequate:
                print("  → Generating improved alt text suggestion...")
                suggestion = generate_alt_text_suggestion(llm, image_path)
                print(f"  → Suggestion: {suggestion.get('suggested_alt_text', '')[:80]}...")

            # Store results
            result_row = {
                'image_path': image_path,
                'current_alt_text': current_alt,
                'status': 'ADEQUATE' if is_adequate else 'NEEDS_IMPROVEMENT',
                'is_adequate': is_adequate,
                'accuracy_score': accuracy,
                'completeness_score': completeness,
                'accessibility_score': accessibility,
                'issues': '; '.join(validation.get('issues', [])),
                'overall_assessment': validation.get('overall_assessment', ''),
                'suggested_alt_text': suggestion.get('suggested_alt_text', '') if suggestion else '',
                'detailed_description': suggestion.get('detailed_description', '') if suggestion else '',
                'image_type': suggestion.get('image_type', '') if suggestion else '',
                'reasoning': suggestion.get('reasoning', '') if suggestion else ''
            }

            results.append(result_row)

    # Write results to output CSV
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    output_fields = [
        'image_path',
        'current_alt_text',
        'status',
        'is_adequate',
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
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Results saved to: {output_file}")

    # Print summary
    adequate_count = sum(1 for r in results if r['is_adequate'])
    needs_improvement = len(results) - adequate_count
    error_count = sum(1 for r in results if r.get('status') == 'ERROR')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(results)}")
    print(f"✓ Adequate alt text: {adequate_count}")
    print(f"✗ Needs improvement: {needs_improvement}")
    if error_count > 0:
        print(f"⚠ Errors (file not found): {error_count}")
    print(f"\nAverage scores:")

    valid_results = [r for r in results if r['status'] != 'ERROR']
    if valid_results:
        avg_accuracy = sum(r['accuracy_score'] for r in valid_results) / len(valid_results)
        avg_completeness = sum(r['completeness_score'] for r in valid_results) / len(valid_results)
        avg_accessibility = sum(r['accessibility_score'] for r in valid_results) / len(valid_results)

        print(f"  Accuracy: {avg_accuracy:.1f}/10")
        print(f"  Completeness: {avg_completeness:.1f}/10")
        print(f"  Accessibility: {avg_accessibility:.1f}/10")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NuMarkdown-8B Alt Text Validator")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output-dir", help="Directory to save the output CSV. Defaults to input directory.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Configure global GPU layers based on args
    global GPU_LAYERS
    if args.cpu:
        GPU_LAYERS = 0
        
    input_csv = args.input
    
    # Determine output file
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(input_csv))
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"alttext_validation_{timestamp}.csv"
    output_csv = os.path.join(output_dir, output_filename)

    print("\n" + "=" * 60)
    print("NuMarkdown-8B Alt Text Validator")
    print("=" * 60)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Model cache: {MODEL_CACHE_DIR}")
    print("=" * 60)

    # Step 1: Download/load cached models
    text_model, image_encoder = download_models()

    # Step 2: Initialize model
    llm = initialize_model(text_model, image_encoder)

    # Step 3: Process CSV
    process_csv(llm, input_csv, output_csv)

    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()