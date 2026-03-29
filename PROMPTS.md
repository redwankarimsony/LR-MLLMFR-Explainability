# Prompt Templates

This file documents all system and user prompt templates used across the explanation generation scripts (`Notebook_OpenAi.py`, `Notebook_Gemini.py`, `Notebook_Claude.py`).

We adopt a **multi-level prompting strategy** to evaluate explanation reliability under varying levels of auxiliary face recognition (FR) information. Prompts provide progressively increasing information from FR models.

---

## System Prompts

Two system prompt variants are used across the scripts.

---

<a name="system-prompt-a"></a>

### System Prompt A — Natural Language Style
**Used by:** `Notebook_OpenAi.py` (`system_prompt2`), `Notebook_Claude.py` (`SYSTEM_PROMPT_2`)

```
You are an expert biometric analyst specializing in explainable face comparison.
Given two face images, your goal is to generate a clear, attribute-rich explanation
of their similarities and differences.

You must:
- Compare the two faces across both stable (identity-related) and variable (contextual) features.
- Use natural language sentences rather than rigid bullet templates.
- Mention key biometric attributes where relevant:
  facial structure, eye configuration, nose morphology, mouth and lips, eyebrows, skin tone,
  texture, symmetry, periocular region, ears, hairline, and contextual factors like pose,
  lighting, or expression.
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
```

---

### System Prompt B — Structured Attribute Style
**Used by:** `Notebook_Gemini.py` (`system_prompt`)

```
You are an expert biometric analyst specializing in facial recognition explainability.
Your task is to compare two face images and provide a structured, detailed, and objective explanation
of their visual similarities and differences. Do not say you can't identify the individuals;
focus solely on comparing visual facial features

Follow these rules carefully:
- Use a consistent attribute–value format with section headers exactly as shown below.
- Focus only on visible, identity-relevant features (ignore background, clothing, and accessories
  unless occluding the face).
- Describe both similarities and differences clearly and concisely.
- If an attribute is not visible or uncertain, write "Not clearly visible".
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
```

---

## User Prompt Templates

User prompts are selected based on the prompting strategy (experiment) and the dataset split (training vs. test). The `[Model Scores]` section is appended dynamically at runtime with a JSON block of FR model scores and/or decisions.

---

### i) Grounded Prompting — Training Data Generation Only

Ground-truth labels are explicitly provided to the MLLM. Used **only during training data generation** on BUPT-CBFace. Two variants exist depending on the pair label.

**Genuine Pairs** (`user_prompt_datagen_gen`)
```
Compare the following two face images and generate a detailed explanation of their similarities
and differences following the required format. It is confirmed that those two face images belong
to the same person. The scores of the various face recognition models for this pair are also
provided for your reference. Now provide the comparison focusing on the similarities and differences
across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

[Model Scores]
{ ... }
```

**Impostor Pairs** (`user_prompt_datagen_imp`)
```
Compare the following two face images and generate a detailed explanation of their similarities
and differences following the required format. It is confirmed that those two face images belong
to different persons. The scores of the various face recognition models for this pair are also
provided for your reference. Now provide the comparison focusing on the similarities and differences
across all key facial attributes.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

[Model Scores]
{ ... }
```

> **Note:** In `Notebook_Gemini.py`, the impostor variant reads *"does not belong to the same person"* instead of *"belong to different persons"*. The meaning is equivalent.

---

### ii) No-score Prompting

Only the face image pair is provided. The MLLM generates explanations based **solely on visual evidence**, with no FR model information.

**Variable name:** `user_prompt_dataget_plain_text` (OpenAI, Gemini) / `USER_PROMPT_PLAIN_TEXT` (Claude)
**Experiment name:** `with-no-info`

```
Compare the following two face images and generate a detailed explanation of their similarities
and differences following the required format. Now provide the comparison focusing on the
similarities and differences across all key facial attributes. You must explain in terms of
the visual features following the given format. Do not say you can not. Just do it based on
what you see.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>
```

---

### iii) Score-only Prompting

The MLLM receives the face image pair and FR similarity scores **without** binary match/non-match decisions. Two sub-variants exist.

<a name="prompt-iii-a"></a>

#### iii-a) Multi-model Scores
**Variable name:** `user_prompt_dataget_inference_with_multi_scores` (OpenAI, Gemini) / `USER_PROMPT_WITH_SCORES` (Claude)
**Experiment name:** `with-scores`
**FR models provided:** ArcFace, CosFace, AdaFace, MagFace, FaceNet-VGGFace2, FaceNet-CasiaWebFace, KPRPE

```
Compare the following two face images and generate a detailed explanation of their similarities
and differences following the required format. The scores of the various face recognition models
in a scale of [0, 1] for this pair are also provided for your reference. Now provide the
comparison focusing on the similarities and differences across all key facial attributes.
You must explain in terms of the visual features rather than the scores provided.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

[Model Scores]
{ ... }
```

#### iii-b) Single-model Score + Decision (KPRPE)
**Variable name:** `user_prompt_dataget_inference_single_model_score` (OpenAI, Gemini) / `USER_PROMPT_SINGLE_MODEL_SCORE` (Claude)
**Experiment name:** `with-kprpe-score-decision`
**FR model provided:** KPRPE only

```
Compare the following two face images and generate a detailed explanation of their similarities
and differences following the required format. The score of a face recognition model in a
scale of [0, 1] and its decision for the given pair is also provided for your reference. Now
provide the comparison focusing on the similarities and differences across all key facial
attributes. You must explain in terms of the visual features rather than the scores provided.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

[Model Scores]
{ ... }
```

---

<a name="prompt-iv"></a>

### iv) Score+Decision Prompting

The MLLM receives the face image pair, FR similarity scores, **and** binary match/non-match decisions thresholded at 0.01% FMR.

**Variable name:** `user_prompt_dataget_inference_with_scores_decisions` (OpenAI, Gemini) / `USER_PROMPT_WITH_SCORES_DECISIONS` (Claude)
**Experiment name:** `with-scores-decisions`
**FR models provided:** ArcFace, CosFace, AdaFace, MagFace, FaceNet-VGGFace2, FaceNet-CasiaWebFace, KPRPE

```
Compare the following two face images and generate a detailed explanation of their similarities
and differences following the required format. The scores of the various face recognition models
and their decision for the given pair are also provided for your reference. Now provide the
comparison focusing on the similarities and differences across all key facial attributes.
You must explain in terms of the visual features rather than the scores provided.

Image 1: <image_1_placeholder>
Image 2: <image_2_placeholder>

[Model Scores]
{ ... }
```

---

## Summary Table

| Strategy | Experiment Name | GT Label | FR Scores | FR Decisions | Split |
|---|---|---|---|---|---|
| i) Grounded | *(training only)* | Yes | Yes (multi) | Yes | Train |
| ii) No-score | `with-no-info` | No | No | No | Test |
| iii-a) Score-only (multi) | `with-scores` | No | Yes (multi) | No | Test |
| iii-b) Score+Dec (KPRPE) | `with-kprpe-score-decision` | No | Yes (single) | Yes | Test |
| iv) Score+Decision (multi) | `with-scores-decisions` | No | Yes (multi) | Yes | Test |

---

## Runtime Prompt Assembly

At runtime, FR model scores (and optionally decisions) are appended to the user prompt as a JSON block:

```python
result_dict = evaluate_row(row, THRESHOLDS, MODELS, include_prediction=True)
dict_json_str = json.dumps(result_dict, indent=2)
user_prompt += f"\n\n[Model Scores]\n{dict_json_str}"
```

Example `[Model Scores]` block (Score+Decision):
```json
{
  "KPRPE": [0.72, "Match"]
}
```

Example `[Model Scores]` block (multi-model, Score+Decision):
```json
{
  "ArcFace": [0.32, "Non-Match"],
  "CosFace": [0.51, "Match"],
  "KPRPE": [0.72, "Match"]
}
```
