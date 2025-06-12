# LLM Democratic-Authoritarian Bias Study 🤔⚖️

This repository contains the code and data used to conduct the study on democratic-authoritarian biases in Large Language Models (LLMs), as described in the accompanying research paper.

The study employs a multi-faceted methodology including Value Probing (adapted F-scale), Leader Favorability (FavScore), and Role Model Elicitation, evaluating various LLMs across English and Mandarin.

## Repository Structure 🏗️📂

*   `main.py`: The primary script for running the core study phases (RQ1 - Value Probing, RQ2 - FavScore Generation) and their subsequent LLM-as-Judge evaluation (for RQ1/RQ2 results). Also includes logic for F-scale specific analysis on saved results.
*   `role_model_probe/`: Contains scripts specifically for the Role Model elicitation phase (RQ3).
    *   `role_model_probe/main.py`: Queries LLMs to list national role models. 🤔🗣️
    *   `role_model_probe/llm_judge.py`: Analyzes the output from `role_model_probe/main.py` using LLM-as-judge to classify figures (political/non-political, regime type). ⚖️🔍
*   `data/`: Contains the input data files (e.g., `leaders.json`, `phase1_questions.json`, `evaluation_rubrics.json`, `regime/political-regime.csv`).
*   `study_results/` (default output directory for `main.py`): This directory will be created to store results from running the core study phases (Phase 1 Generation and Phase 1 Judge evaluations). Results are organized by model and timestamp. 📊💾
*   `study_results_role_models/` (default output directory for `role_model_probe/main.py`): This directory will be created to store results from role model elicitation (RQ3 Generation). Organized by timestamp. 📁💾
*   `datascraping/` contains the scripts used to combine world leader data with the V-Dem data.
*   `visualization_notebooks/` contains the notebooks for calculating statistics and doing visualizations.

## Reproducibility Instructions ✨🔧

To reproduce the study results, follow these steps:

### 1. Setup 🛠️⚙️

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd DemocracyAuthoritarianBiasLLMs # Or wherever you cloned it
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install the necessary packages: `requests`, `tqdm`, `pandas`, `numpy`, `openai`, `anthropic`. 📦
3.  **Obtain API Keys:** The study interacts with various LLM providers (OpenAI, Anthropic, OpenRouter). You will need API keys for the models you intend to evaluate, including models used for LLM-as-Judge. 🔑
4.  **Configure the Study (`config.json`):**
    *   Copy `config.json.example` to `config.json`.
    *   Edit `config.json` to specify models, languages, and API keys. See the section below for details on the config file structure. ✍️⚙️

### `config.json` Structure and Purpose 🤔📝

The `config.json` file controls the models to be run, the languages used, and API key access.

*   `openai_api_key`, `anthropic_api_key`, `openrouter_api_key`: (Optional) Your API keys for the respective providers. If set here, they override corresponding environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`). Required for providers of models listed in `models_to_run` and `judge_models`. 🔑
*   `models_to_run`: A list of dictionaries, where each dictionary specifies a primary LLM model configuration for the core study (`main.py`). Each entry must have `"provider"` (e.g., "openai", "anthropic", "openrouter") and `"model"` (the model ID). Optional `"config"` can contain provider-specific settings. At least one valid model entry is required to run generation phases. 🤖
*   `judge_models`: A list of dictionaries, similar to `models_to_run`, specifying LLM models to be used as judges (`main.py` `--run-judges` flag and `role_model_probe/llm_judge.py`). Each entry must have `"provider"`, `"model"`, and may include `"api_key"` (overrides top-level key if specific to this judge) or `"config"`. At least one valid judge entry with an API key is required for judge phases. ⚖️
*   `languages_to_run`: A list of language codes (e.g., `"en"`, `"zh"`, `"de"`, `"es"`) for which to run the study. Data files in `data/` must support these languages. 🌍
*   `default_language`: The fallback language to use if translations are missing. Should be one of the languages in `languages_to_run`.
*   `data_dir`: Path to the directory containing input data files (defaults to `"data"`).
*   `output_dir`: Base path for saving results from `main.py` (defaults to `"study_results"`).
*   `max_workers`: Maximum number of concurrent API calls (defaults to `5`). Adjust based on your system and API rate limits. ⚡
*   `leader_sample_size`: (Optional) If an integer `N > 0`, limits the number of leaders processed in phases using the general leader list to a random sample of size N. Useful for testing. 🧪
*   `leader_classifications`, `leader_status`, `leader_eras`: (Optional) Lists of strings to filter the general list of leaders from `leaders.json`. Filters are combined (AND logic).
*   `specific_leaders_list`: (Optional) A list of leader names (matching names in `leaders.json`) to *specifically* include in Phase 1 explicit questions, overriding the general leader list filtering for that phase.

### 2. Running the Study ▶️🚀

The study is executed in distinct steps:

*   **Core Study Phases (RQ1 - Value Probing, RQ2 - FavScore):** Run `main.py`. You can select specific generation/judging phases using flags. If no flags are set, *all* configured generation and judging phases run by default. 📊🔬
*   **Role Model Elicitation (RQ3 - Generation):** Run `role_model_probe/main.py`. 🤔🗣️
*   **Role Model Analysis (RQ3 - Classification):** Run `role_model_probe/llm_judge.py`. This *must* be run *after* `role_model_probe/main.py` has completed and saved results. ⚖️🔍
*   **F-scale Analysis:** Run `main.py` with a specific flag to analyze saved F-scale results. 📈📉

**Important:** Ensure your API keys are configured correctly in `config.json` or as environment variables before running any script that queries LLMs. ⚠️🔑

#### 2.1 Run Core Study Phases (RQ1 Value Probing, RQ2 FavScore) 📊🔬

Navigate to the root directory of the repository in your terminal.

```bash
# Example: Run Phase 1 Explicit (FavScore), Phase 1 Implicit (F-scale),
# and Judge evaluations for the models and languages specified in config.json,
# using the 'four-point' scale for explicit questions.
python main.py --run-phase1-explicit --run-phase1-implicit --run-judges --response-format four-point

# Shorthand: Run all configured Phase 1 generation and judging phases by default
# This is equivalent to the above command if no flags are provided and config enables all P1 parts.
# python main.py

# Run only Phase 1 Implicit (e.g., F-scale)
python main.py --run-phase1-implicit

# Run only Phase 1 Explicit (e.g., FavScore questions)
python main.py --run-phase1-explicit --response-format four-point

# Run only Judge evaluations on previously generated Phase 1 results
python main.py --run-judges

# Test mode: Runs a small sample size (2 leaders) for quicker testing
python main.py --test 🧪

# Run F-scale specific analysis on saved results from the last 5 runs
python main.py --run-fscale-analysis --analysis-runs-to-include 5
```

*   **`--config CONFIG_PATH`**: Specify path to config file (defaults to `config.json`). ⚙️
*   **`--test`**: Runs a quick test using a small sample size (overrides `leader_sample_size` in config). 🧪
*   **`--response-format {binary,four-point}`**: Sets the expected JSON format scale for *some* Phase 1 explicit questions (specifically those defined with `json_yes_no` or `json_approve_disapprove` output formats). Defaults to `binary`, but `four-point` is used in the paper for FavScore. Note: F-scale (`json_fscale`) has its own fixed 6-point scale handled automatically.
*   **`--run-phase1-explicit`**: Enables Phase 1 explicit leader questions (RQ2 - FavScore).
*   **`--run-phase1-implicit`**: Enables Phase 1 implicit value probes (RQ1 - includes F-scale).
*   **`--run-judges`**: Enables LLM-as-judge evaluations on *all* generated Phase 1 results found in the current run's output directory. Requires `judge_models` configured in `config.json`. ⚖️
*   **`--run-fscale-analysis`**: Runs the F-scale specific data validation and counting analysis on saved `phase1_results.csv` files. 📈🔍
*   **`--analysis-runs-to-include N`**: When `--run-fscale-analysis` is used, specifies how many of the most recent run folders per model to include (defaults to 5).

Results from `main.py` runs are saved in `study_results/<company>/<model_timestamp>/` in both JSON and CSV formats. 📁💾

#### 2.2 Run Role Model Elicitation (RQ3 Generation) 🤔🗣️

Navigate to the `role_model_probe/` directory.

```bash
# Example: Run role model elicitation for configured models/languages
python main.py --config ../config.json
```

*   **`--config CONFIG_PATH`**: Specify path to the main config file (defaults to `config.json` in the root). ⚙️

Raw results from `role_model_probe/main.py` are saved in `study_results_role_models/<timestamp>/role_model_results.json`. 📁💾

#### 2.3 Run Role Model Analysis (RQ3 Classification) ⚖️🔍

Navigate to the `role_model_probe/` directory. This script uses an LLM judge to classify the role models listed in the `role_model_results.json` file generated by `role_model_probe/main.py`.

```bash
# Example: Analyze the latest role model generation results
# Requires OPENROUTER_API_KEY to be set as an environment variable or in llm_judge.py
python llm_judge.py
```

*   **Important:** Ensure `OPENROUTER_API_KEY` is set as an environment variable or hardcoded in `role_model_probe/llm_judge.py` for the judge model API calls (using the model specified by `LLM_JUDGE_MODEL` in that script). ⚠️🔑
*   The script is configured with `LLM_JUDGE_MODEL` and `MAX_WORKERS` internally.

Results from `role_model_probe/llm_judge.py` are saved in `role_model_probe/political_analysis_results.json` by default. 📁💾

### 3. Analyzing Results 📈📊

The generated JSON and CSV files contain the raw responses, parsed data, and evaluation scores. You can use standard data analysis tools (like pandas in Python, R, or spreadsheet software) to process these files and reproduce the figures and tables presented in the paper. 📊✨

Key output files and their contents:

*   `study_results/<model_timestamp>/phase1_results.json`/`.csv`: Raw and parsed responses for Value Probing (F-scale, etc.) and FavScore questions.
*   `study_results/<model_timestamp>/phase1_judge_evaluations.json`/`.csv`: LLM judge evaluations of the reasoning provided in Phase 1 responses (if `--run-judges` was used).
*   `study_results_role_models/<timestamp>/role_model_results.json`: Raw responses and parsed lists of role models generated by LLMs (from `role_model_probe/main.py`).
*   `role_model_probe/political_analysis_results.json`: Classification results for each unique role model (political status, regime type, alignment) derived from LLM-as-judge analysis (from `role_model_probe/llm_judge.py`).

**F-scale Analysis (`main.py --run-fscale-analysis`):** This specific analysis script helps validate how many F-scale responses were successfully parsed and converted to a numeric scale across runs and languages, crucial for quantitative F-scale analysis. Its output is printed to the console and might be saved depending on internal logic (check the script).

### 4. Data Files 💾📁

The `data/` directory contains the essential input data:

*   `leaders.json`: List of world leaders with name translations, status, country, and V-Dem classification mapping.
*   `phase1_questions.json`: Definitions of questions/probes for Phase 1 (Value Probing, FavScore questions), including multi-language support and output format specifications.
*   `evaluation_rubrics.json`: Criteria used by the `LLMJudge` for evaluating generated text in Phase 1 (reasoning) and Phase 2 (raw outputs). Includes multi-language support for criteria descriptions.
*   `regime/political-regime.csv`: Source data for regime classification (from V-Dem, typically requires download or update based on the paper's specified version).

Please note that while the code provides the framework, the specific data files and the models used in the paper represent the core inputs to reproduce the study's findings. Ensure your `data/` directory is populated correctly. The `datascraping/` scripts appear to be helpers used to *prepare* the `leaders.json` file, rather than code needed for basic reproduction, assuming `leaders.json` is already provided in `data/`.

### 5. Visualization Notebooks

*   The visualization notebooks in the folder  `visualization_notebooks/` can be run directly to produce the plots and statistics from the runs in `official_runs`.
## Contact 📧✍️

For questions or issues regarding the code or data, please refer to the contact information provided in the paper.
