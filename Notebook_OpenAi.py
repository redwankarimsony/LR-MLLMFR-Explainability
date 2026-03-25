# %%
import json
import argparse
from matplotlib import pyplot as plt
from openai import OpenAI
from run_openai_sequential import sha256_text, load_text, compress_to_data_url, build_messages, to_responses_input, build_messages_gpt52
import pandas as pd
from tqdm import tqdm
import re
import os


# %%
# # Setting and loading the prompts
# PROMPT_DIR = "/mnt/research/iPRoBeLab/sonymd/LikelihoodRatio/text_generators/prompts/"
# EXPERIMENT_NAME = "experiment3_with_label_guided_prompt"

# SYSTEM_PROMPT_PATH = os.path.join(PROMPT_DIR, EXPERIMENT_NAME, "prompt_system.txt")
# USER_PROMPT_PATH_MATCH = os.path.join(PROMPT_DIR, EXPERIMENT_NAME, "prompt_user_match.txt")
# USER_PROMPT_PATH_NONMATCH = os.path.join(PROMPT_DIR, EXPERIMENT_NAME, "prompt_user_nonmatch.txt")


# system_prompt = load_text(SYSTEM_PROMPT_PATH) + "\n Remember you do not have to identify the persons in the images, just compare the two face images provided."
# user_prompt = load_text(USER_PROMPT_PATH_MATCH)

# %%
system_prompt = '''You are an expert biometric analyst specializing in facial recognition explainability. 
Your task is to compare two face images and provide a structured, detailed, and objective explanation 
of their visual similarities and differences. Do not say you can’t identify the individuals; 
focus solely on comparing visual facial features

Follow these rules carefully:
- Use a consistent attribute–value format with section headers exactly as shown below.
- Focus only on visible, identity-relevant features (ignore background, clothing, and accessories unless occluding the face).
- Describe both similarities and differences clearly and concisely.
- If an attribute is not visible or uncertain, write “Not clearly visible”.
- End with a logical reasoning statement summarizing whether the faces likely belong to the same individual.

Use the following output structure verbatim:

[Face Matching Explanation]

Match Verdict: (Match / Non-Match / Uncertain)

Similarity Attributes:
- Facial Structure:
- Eye Configuration:
- Nose Morphology:
- Mouth and Lip Geometry:
- Eyebrow Structure:
- Facial Symmetry and Proportions:
- Texture and Tone Consistency:
- Periocular Region Details:
- Ear Shape:
- Hairline and Forehead Ratio:

Distinctive Differences:
- (List all visible differences or contextual variations such as pose, lighting, expression, or occlusion.)

Overall Reasoning:
(Provide a concise conclusion integrating the above cues to justify the verdict.)
'''


user_prompt = '''Compare the following two face images and generate a detailed explanation following the given format.

[Instruction]
Analyze both images and describe the similarities and differences across all key facial attributes. 
Base your judgment solely on visual evidence from the faces. 
Then, decide whether they likely belong to the same person or to different individuals.

[Images]
Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

[Output Format Reminder]
Follow the structure below exactly:
- Match Verdict
- Similarity Attributes (with all sub-attributes)
- Distinctive Differences
- Overall Reasoning'''


system_prompt2 = '''You are an expert biometric analyst specializing in explainable face comparison. 
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

Do not say you can’t identify the individuals; focus solely on comparing visual facial features
'''


user_prompt2 = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The scores of the various face recognition models for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""


user_prompt_datagen_gen = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. It is confirmed that those two face images belong to the same person. The scores of the various face recognition models for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""

user_prompt_datagen_imp = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. It is confirmed that those two face images belong to different persons. The scores of the various face recognition models for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""


user_prompt_dataget_plain_text = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format.  Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features following the given format. Do not say you can not. Just do it based on what you see. 

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""



user_prompt_dataget_inference_with_multi_scores = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The scores of the various face recognition models in an scale of [0, 1] for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features rather than the scores provided. 

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""

user_prompt_dataget_inference_with_scores_decisions = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The scores of the various face recognition models and their decision for the given pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features rather than the scores provided. 

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
"""



user_prompt_dataget_inference_single_model_score = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The score of a face recognition model in a scale of [0, 1] and its decision for the given pair is also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes. You must explain in terms of the visual features rather than the scores provided. 

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
"""



def evaluate_row(row, thresholds, models, include_prediction=False):
    """Evaluate a row of model scores against thresholds.
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



# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# IMAGES_DIR = ".data/BUPT-CBFace-12-top-100/images"
# METADATA_FILE = ".data/BUPT-CBFace-12-top-100/cbface_top100_pairs_scores.csv"
# OUTPUT_DIR = ".data/BUPT-CBFace-12-top-100/Explanations"

IMAGES_DIR = ".data/IJBS/IJBS-Still"
METADATA_FILE = ".data/IJBS/ijbs_still_benchmark_scores_with_roc.csv"
RESULTS_FILE = ".data/IJBS/ijbs_still_benchmark_results.csv"
# MODELS = ['ArcFace', 'CosFace', 'AdaFace', "MagFace", "FaceNet_vggface2", "FaceNet_casia_webface", "KPRPE"]
MODELS = ["KPRPE",]
OUTPUT_DIR = ".data/IJBS/Explanations-with-scores-gt"
# MODEL_NAME = "gpt-4o"
MODEL_NAME = "gpt-5.2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing')
parser.add_argument('--end_idx', type=int, default=100000, help='End index for processing')
args = parser.parse_args()

START_IDX = args.start_idx
END_IDX = args.end_idx

df = pd.read_csv(METADATA_FILE)

END_IDX = min(END_IDX, len(df))
df = df.iloc[START_IDX:END_IDX]
# Select one row from the dataframe



# Filter the dataframe for processing based on existing outputs
unprocessed_rows = []
for index, row in df.iterrows():
    pair_id = row['pair_id']
    output_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pair_id}.txt")
    if not os.path.exists(output_path):
        unprocessed_rows.append(row)

df = pd.DataFrame(unprocessed_rows)
print(f"Total unprocessed rows: {len(df)}")
results_df = pd.read_csv(RESULTS_FILE)
results_df.set_index('Model', inplace=True)

THRESHOLDS = {
    model: results_df.loc[model, "THR@FMR=0.01%"]
    for model in MODELS
}


output_str = f"Idx {START_IDX} to {END_IDX - 1}"


for index, row in tqdm(df.iterrows(), total=len(df), desc=output_str):
    # Extract the pair_id and check if output already exists
    pair_id = row['pair_id']
    output_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pair_id}.txt")
    if os.path.exists(output_path):
        print(f"Output for pair_id {pair_id} already exists. Skipping...")
        continue

    # Get image paths
    img1_filename = row['image1']
    img2_filename = row['image2']
    img1_path = os.path.join(IMAGES_DIR, img1_filename)
    img2_path = os.path.join(IMAGES_DIR, img2_filename)


    if 'IJBS' in IMAGES_DIR:        # For IJBS dataset, we do not provide the label information just the scores. 
        # user_prompt = user_prompt_dataget_inference_single_model_score    # For the single model score and decision
        # user_prompt = user_prompt_dataget_inference_with_multi_scores       # For the multi scores only # make sure you put prediction false.
        user_prompt = user_prompt_dataget_inference_with_scores_decisions 
        # user_prompt = user_prompt_dataget_plain_text
    else:
        if row['label'] == 1:
            user_prompt = user_prompt_datagen_gen
        else:
            user_prompt = user_prompt_datagen_imp

    

    result_dict = evaluate_row(row, THRESHOLDS, MODELS, include_prediction=True)
    dict_json_str = json.dumps(result_dict, indent=2)
    user_prompt += f"\n\n[Model Scores]\n{dict_json_str}"
    # Convert dictionary to JSON string for inclusion in the prompt
    

    


    url1 = compress_to_data_url(img1_path, fmt="JPEG", max_side=160, quality=85)
    url2 = compress_to_data_url(img2_path, fmt="JPEG", max_side=160, quality=85)

    # Build messages for OpenAI API

    # Call OpenAI API with retry logic
    RETRY_LIMIT = 5

    while RETRY_LIMIT:
        if MODEL_NAME == "gpt-4o":
            message = build_messages(system_prompt2, user_prompt, url1, url2)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=message,
                temperature=0.7,
                max_tokens=2000,
            )
            text_response = response.choices[0].message.content

        elif MODEL_NAME == "gpt-5.2":
            message = build_messages_gpt52(system_prompt2, user_prompt, url1, url2)
            response=client.responses.create(
                model=MODEL_NAME,
                input=message,
                temperature=0.7,
                max_output_tokens=1200,
            )
            text_response = response.output[0].content[0].text
        # response = client.chat.completions.create(
        #     model=MODEL_NAME,
        #     messages=message,
        #     temperature=0.7,
        #     max_tokens=2000,
        # )

        if text_response is None:
            continue

        # Count number of lines in the response
        num_lines = text_response.count('\n') + 1
        RETRY_LIMIT -= 1

        headers = ['Similarities', "Differences", "Overall Reasoning"]
        if all(header in text_response for header in headers):
            break
        else:
            continue
        

    if num_lines <= 2:
        print(f"Warning: Response for pair_id {pair_id} seems too short ({num_lines} lines).")
        print("Response content:")
        print(text_response)
    else:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(text_response)
            f.flush()

# %%
