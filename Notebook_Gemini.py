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
of their visual similarities and differences. Do not say you can’t identify the individuals; focus solely on comparing visual facial features

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


user_prompt2 = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. The scores of the various face recognition models for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""


user_prompt_datagen_gen = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. It is confirmed that those two face images belong to the same person. The scores of the various face recognition models for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

"""

user_prompt_datagen_imp = """Compare the following two face images and generate a detailed explanation of their similarities and differences following the required format. It is confirmed that those two face images does not belong to the same person. The scores of the various face recognition models for this pair are also provided for your reference. Now provide the comparison focusing on the similarities and differences across all key facial attributes.

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



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE COPY FROM THE OTHER SCRIPT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



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

def load_image_as_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()




# Initialize OpenAI client
client = genai.Client()

# IMAGES_DIR = ".data/BUPT-CBFace-12-top-100/images"
# METADATA_FILE = ".data/BUPT-CBFace-12-top-100/cbface_top100_pairs_scores.csv"
# OUTPUT_DIR = ".data/BUPT-CBFace-12-top-100/Explanations"

IMAGES_DIR = ".data/IJBS/IJBS-Still"
METADATA_FILE = ".data/IJBS/ijbs_still_benchmark_scores.csv"
RESULTS_FILE = ".data/IJBS/ijbs_still_benchmark_results.csv"
MODELS = ['ArcFace', 'CosFace', 'AdaFace', "MagFace", "FaceNet_vggface2", "FaceNet_casia_webface", "KPRPE"]
OUTPUT_DIR = ".data/IJBS/Explanations-with-scores-gt"
MODEL_NAME = "gemini-2.5-flash"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing')
parser.add_argument('--end_idx', type=int, default=1000000, help='End index for processing')
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
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
for index, row in tqdm(df.iterrows(), total=len(df), desc=output_str):

    # ======= Progressive Backup: Skip if output exists =======
    pair_id = row['pair_id']
    output_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pair_id}.txt")
    error_path = output_path + ".error"

    if os.path.exists(output_path):
        # print(f"[SKIP] Pair {pair_id} already completed.")
        continue

    # If an error file exists, skip or retry depending on your choice.
    # For now, we skip to avoid repeated failure.
    if os.path.exists(error_path):
        # print(f"[SKIP] Pair {pair_id} previously failed.")
        continue

    # ======= Load Image Bytes =======
    img1_path = os.path.join(IMAGES_DIR, row['image1'])
    img2_path = os.path.join(IMAGES_DIR, row['image2'])

    img1_bytes = load_image_as_bytes(img1_path)
    img2_bytes = load_image_as_bytes(img2_path)

    # ======= Build Prompt =======
    if 'IJBS' in IMAGES_DIR:
        user_prompt = user_prompt_dataget_inference2
    else:
        user_prompt = user_prompt_datagen_gen if row['label'] == 1 else user_prompt_datagen_imp

    result_dict = evaluate_row(row, THRESHOLDS, MODELS, include_prediction=True)
    dict_json_str = json.dumps(result_dict, indent=2)

    # ========== Retry Loop ==========
    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",

                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),

                contents=[
                    user_prompt,
                    types.Part.from_bytes(data=img1_bytes, mime_type="image/jpeg"),
                    types.Part.from_bytes(data=img2_bytes, mime_type="image/jpeg"),
                ]
            )

            text_response = (response.text or "").strip()

            # ========== Successful write ==========
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_response)

            break  # Exit retry loop and go to next row

        except errors.ServerError as e:
            msg = str(e)

            # ======= Overload / UNAVAILABLE (retryable) =======
            if "503" in msg or "UNAVAILABLE" in msg:
                attempt += 1

                if attempt > MAX_RETRIES:
                    # Write an error log and move on
                    with open(error_path, "w", encoding="utf-8") as ef:
                        ef.write(f"FAILED after {MAX_RETRIES} retries:\n{msg}\n")
                    break

                delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1.0)
                # print(f"[RETRY {attempt}] 503 error, waiting {delay:.1f}s")
                time.sleep(delay)
                continue  # retry again

            # ======= Non-retryable server error =======
            with open(error_path, "w", encoding="utf-8") as ef:
                ef.write(f"NON-RETRYABLE ERROR:\n{msg}\n")
            break

