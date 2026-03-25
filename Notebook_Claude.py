"""
Face Comparison Explanation Generator using Claude API

This script processes face image pairs and generates detailed explanations
of their visual similarities and differences using the Claude API.
"""

import json
import argparse
import pandas as pd
from tqdm import tqdm
import os
import io
from typing import List, Dict, Any, Optional
import base64

from anthropic import Anthropic
from run_openai_sequential import (
    compress_to_data_url,
    build_messages,
    build_messages_gpt52,
)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def filter_unprocessed_rows(
    metadata_file: str,
    output_dir: str,
    model_name: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load metadata CSV, slice by indices, and return only rows
    whose output files do not yet exist.

    Args:
        metadata_file: Path to CSV file
        output_dir: Base output directory
        model_name: Subdirectory for model outputs
        start_idx: Starting index for slicing
        end_idx: Ending index for slicing (optional)

    Returns:
        DataFrame containing only unprocessed rows
    """
    df = pd.read_csv(metadata_file)

    # Handle slicing safely
    if end_idx is not None:
        end_idx = min(end_idx, len(df))
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]

    # Prepare output directory once
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Filter rows
    unprocessed_rows = []
    for _, row in df.iterrows():
        pair_id = row["pair_id"]
        output_path = os.path.join(model_output_dir, f"{pair_id}.txt")

        if not os.path.exists(output_path):
            unprocessed_rows.append(row)

    return pd.DataFrame(unprocessed_rows)


def get_thresholds(results_file: str, models: List[str]) -> Dict[str, float]:
    """
    Load the thresholds for the models from the results file.

    Args:
        results_file: Path to results CSV file
        models: List of model names

    Returns:
        Dictionary mapping model names to thresholds
    """
    results_df = pd.read_csv(results_file)
    results_df.set_index('Model', inplace=True)
    thresholds = {
        model: results_df.loc[model, "THR@FMR=0.01%"]
        for model in models
    }
    return thresholds


def evaluate_row(
    row: pd.Series,
    thresholds: Dict[str, float],
    models: List[str],
    include_prediction: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a row of model scores against thresholds.

    Args:
        row: DataFrame row with model scores
        thresholds: Dictionary of model thresholds
        models: List of model names to evaluate
        include_prediction: Whether to include Match/Non-Match prediction

    Returns:
        Dictionary with evaluation results
    """
    res = {}
    for model in models:
        score = row[model]
        status = "Match" if score >= thresholds[model] else "Non-Match"
        if include_prediction:
            res[model] = (round(score, 2), status)
        else:
            res[model] = round(score, 2)
    return res


def build_messages_claude(
    system_prompt: str,
    user_prompt: str,
    image_path1: str,
    image_path2: str,
    media_type: str = "image/jpeg",
) -> Dict[str, Any]:
    """
    Build Claude-compatible message payload with two images.

    Args:
        system_prompt: System instruction
        user_prompt: User query
        image_path1: Path to first image
        image_path2: Path to second image
        media_type: image/jpeg or image/png

    Returns:
        Dictionary with system and messages for Claude API
    """
    def encode_image(path):
        from PIL import Image
        with Image.open(path) as img:
            img.thumbnail((160, 160))
            buffer = io.BytesIO()
            img.save(buffer, format=media_type.split('/')[1].upper(), quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    base64_image1 = encode_image(image_path1)
    base64_image2 = encode_image(image_path2)

    return {
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image1,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image2,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ],
    }


# =====================================================================
# SYSTEM PROMPTS
# =====================================================================

SYSTEM_PROMPT_2 = '''You are an expert biometric analyst specializing in explainable face comparison. 
Given two face images, your goal is to generate a clear, attribute-rich explanation 
of their similarities and differences.

You must:
- Compare the two faces across both stable (identity-related) and variable (contextual) features.  
- Use natural language sentences rather than rigid bullet templates.  
- Mention key biometric attributes where relevant:
  facial structure, eye configuration, nose morphology, mouth and lips, eyebrows, skin tone, texture, symmetry, periocular region, ears, hairline, and contextual factors like pose, lighting, or expression.
- Feel free to add other relevant features if noticeable (e.g., wrinkles, dimples, scars, moles).
- Write in a consistent, professional tone, with one paragraph for similarities and one for differences.
- Conclude with a reasoned verdict: Match / Non-Match / Uncertain.

Use this output format:

[Face Matching Explanation]

Match Verdict: (Match / Non-Match / Uncertain)

Similarities:
(Describe observed facial similarities in natural, descriptive language.)

Differences:
(Describe noticeable differences or contextual variations.)

Overall Reasoning:
(Explain how these cues support your verdict.)

Do not say you can't identify the individuals; focus solely on comparing visual facial features
'''


# =====================================================================
# USER PROMPT TEMPLATES
# =====================================================================

USER_PROMPT_PLAIN_TEXT = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features following the given format. Do not say you can not. Just do it based on what you see.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
"""

USER_PROMPT_WITH_SCORES = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The scores of the various face recognition models in a scale of [0, 1] for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features rather than the scores provided.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
"""

USER_PROMPT_WITH_SCORES_DECISIONS = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The scores of the various face recognition models and their decision for the given pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features rather than the scores provided.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
"""

USER_PROMPT_SINGLE_MODEL_SCORE = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The score of a face recognition model in a scale of [0, 1] and its decision for the given pair is also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features rather than the scores provided.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
"""


# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

EXPERIMENT_CONFIG = {
    'with-no-info': {
        'models': [],
        'user_prompt': USER_PROMPT_PLAIN_TEXT,
    },
    'with-scores': {
        'models': ['ArcFace', 'CosFace', 'AdaFace', 'MagFace', 'FaceNet_vggface2', 'FaceNet_casia_webface', 'KPRPE'],
        'user_prompt': USER_PROMPT_WITH_SCORES,
    },
    'with-scores-decisions': {
        'models': ['ArcFace', 'CosFace', 'AdaFace', 'MagFace', 'FaceNet_vggface2', 'FaceNet_casia_webface', 'KPRPE'],
        'user_prompt': USER_PROMPT_WITH_SCORES_DECISIONS,
    },
    'with-kprpe-score-decision': {
        'models': ['KPRPE'],
        'user_prompt': USER_PROMPT_SINGLE_MODEL_SCORE,
    },
}

DATASET_CONFIG = {
    'IJBS': {
        'images_dir': '.data/IJBS/IJBS-Still',
        'metadata_file': '.data/IJBS/ijbs_still_benchmark_scores_with_roc.csv',
        'results_file': '.data/IJBS/ijbs_still_benchmark_results.csv',
    },
}


# =====================================================================
# ARGUMENT PARSING AND CONFIGURATION
# =====================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate face comparison explanations using Claude API'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Start index for processing'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=100000,
        help='End index for processing'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='claude-opus-4-6',
        help='Model name for generating explanations'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='with-kprpe-score-decision',
        choices=list(EXPERIMENT_CONFIG.keys()),
        help='Experiment name for output directory organization'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='IJBS',
        choices=list(DATASET_CONFIG.keys()),
        help='Dataset name for input configuration'
    )
    parser.add_argument(
        '--retry_limit',
        type=int,
        default=5,
        help='Number of retries for API calls'
    )
    return parser.parse_args()


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Validate experiment and dataset configurations
    if args.experiment_name not in EXPERIMENT_CONFIG:
        raise ValueError(f"Invalid experiment name: {args.experiment_name}")
    if args.dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    # Get configuration
    exp_config = EXPERIMENT_CONFIG[args.experiment_name]
    dataset_config = DATASET_CONFIG[args.dataset_name]

    # Setup paths and directories
    images_dir = dataset_config['images_dir']
    metadata_file = dataset_config['metadata_file']
    results_file = dataset_config['results_file']
    output_dir = f".data/{args.dataset_name}/Explanations-{args.experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load and filter data
    df = filter_unprocessed_rows(metadata_file, output_dir, args.model_name, args.start_idx, args.end_idx)
    thresholds = get_thresholds(results_file, exp_config['models'])

    print(f"Total unprocessed rows: {len(df)}")

    # Process each row
    output_str = f"Idx {args.start_idx} to {args.end_idx - 1}"
    for _, row in tqdm(df.iterrows(), total=len(df), desc=output_str):
        pair_id = row['pair_id']
        result_output_dir = os.path.join(output_dir, args.model_name)
        os.makedirs(result_output_dir, exist_ok=True)
        output_path = os.path.join(result_output_dir, f"{pair_id}.txt")

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Output for pair_id {pair_id} already exists. Skipping...")
            continue

        # Get image paths
        img1_filename = row['image1']
        img2_filename = row['image2']
        img1_path = os.path.join(images_dir, img1_filename)
        img2_path = os.path.join(images_dir, img2_filename)

        # Build user prompt with model scores if applicable
        user_prompt = exp_config['user_prompt'].strip()

        if exp_config['models']:
            result_dict = evaluate_row(
                row,
                thresholds,
                exp_config['models'],
                include_prediction=(args.experiment_name in ['with-scores-decisions', 'with-kprpe-score-decision'])
            )
            dict_json_str = json.dumps(result_dict, indent=2)

            if args.experiment_name == 'with-scores':
                user_prompt += f"\n\n[Model Scores]\n{dict_json_str}"
            elif args.experiment_name == 'with-scores-decisions':
                user_prompt += f"\n\n[Model Scores and Decisions]\n{dict_json_str}"
            elif args.experiment_name == 'with-kprpe-score-decision':
                user_prompt += f"\n\n[KPRPE Score and Decision]\n{dict_json_str}"

        # Prepare image data
        url1 = compress_to_data_url(img1_path, fmt="JPEG", max_side=160, quality=85)
        url2 = compress_to_data_url(img2_path, fmt="JPEG", max_side=160, quality=85)

        # Call API with retry logic
        retry_limit = args.retry_limit
        text_response = None

        while retry_limit > 0:
            try:
                if args.model_name == "gpt-4o":
                    message = build_messages(SYSTEM_PROMPT_2, user_prompt, url1, url2)
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=message,
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    text_response = response.choices[0].message.content

                elif args.model_name == "gpt-5.2":
                    message = build_messages_gpt52(SYSTEM_PROMPT_2, user_prompt, url1, url2)
                    response = client.responses.create(
                        model=args.model_name,
                        input=message,
                        temperature=0.7,
                        max_output_tokens=1200,
                    )
                    text_response = response.output[0].content[0].text

                elif args.model_name.startswith('claude'):
                    message = build_messages_claude(SYSTEM_PROMPT_2, user_prompt, img1_path, img2_path)
                    response = client.messages.create(
                        model=args.model_name,
                        max_tokens=1000,
                        system=message["system"],
                        messages=message["messages"],
                        temperature=0.7
                    )
                    text_response = response.content[0].text

                if text_response is not None:
                    # Check if response contains expected headers
                    headers = ['Similarities', 'Differences', 'Overall Reasoning']
                    if all(header in text_response for header in headers):
                        break

            except Exception as e:
                print(f"Error processing pair_id {pair_id}: {e}")

            retry_limit -= 1

        # Save response if valid
        if text_response is not None:
            num_lines = text_response.count('\n') + 1
            if num_lines <= 2:
                print(f"Warning: Response for pair_id {pair_id} seems too short ({num_lines} lines).")
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(text_response)
                f.flush()
        else:
            print(f"Failed to generate response for pair_id {pair_id} after {args.retry_limit} retries.")


if __name__ == "__main__":
    main()
