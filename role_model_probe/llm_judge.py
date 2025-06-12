def compress_citing_sources(political_analysis_results):
    """Post-process to compress citing sources by combining identical model/language/prompt_type entries."""
    for result in political_analysis_results:
        if not result.get('citing_sources'):
            continue
        
        # Group by model, language, prompt_type
        grouped_sources = {}
        for source in result['citing_sources']:
            key = (source.get('model', ''), source.get('language', ''), source.get('prompt_type', ''))
            if key not in grouped_sources:
                # Create new entry with same structure but with nationalities as a list
                grouped_sources[key] = {
                    'model': source.get('model', ''),
                    'language': source.get('language', ''),
                    'prompt_type': source.get('prompt_type', ''),
                    'nationalities': [source.get('nationality', '')]
                }
            else:
                # Add nationality to existing entry if not already included
                nationality = source.get('nationality', '')
                if nationality and nationality not in grouped_sources[key]['nationalities']:
                    grouped_sources[key]['nationalities'].append(nationality)
        
        # Replace original citing_sources with compressed version
        result['citing_sources'] = list(grouped_sources.values())
    
    return political_analysis_results
def process_role_model(idx, name, nationality, citing_sources, countries, vdem_df):
    """Process a single role model - for use with parallel execution."""
    # Initialize result dictionary
    result = {
        "role_model_name": name,
        "nationality": nationality,
        "countries": countries,  # Add summary of countries this person was mentioned for
        "citing_sources": citing_sources,  # Add citing sources to the result
        "political_classification": None,
        "regime_analysis": None,
        "alignment_analysis": None,
        "errors": []
    }
    
    try:
        # Step 1: Generate and send the political classification prompt
        classification_prompt = generate_political_classification_prompt(name)
        classification_response = query_llm_judge(classification_prompt)
        
        # Sleep to avoid rate limits
        time.sleep(API_CALL_SLEEP)
        
        # Check for errors
        if classification_response.get("error"):
            result["errors"].append(f"Classification Error: {classification_response['error']}")
        
        # Extract the parsed JSON
        classification_json = classification_response.get("parsed_json", {})
        result["political_classification"] = classification_json
        
        # Step 2: If political, proceed with regime analysis using LLM
        if classification_json.get("is_political") == True and classification_json.get("active_period"):
            active_period = classification_json["active_period"]
            
            # Use the LLM to determine the regime type
            regime_type = get_regime_via_llm(name, nationality, active_period, vdem_df)
            result["regime_analysis"] = regime_type
            
            # Sleep to avoid rate limits
            time.sleep(API_CALL_SLEEP)
            
            # Step 3: If regime analysis successful, proceed with alignment analysis
            #if regime_type:
            # Generate and send the alignment prompt
            alignment_prompt = generate_alignment_prompt(name, nationality, active_period, regime_type)
            alignment_response = query_llm_judge(alignment_prompt)
            
            # Sleep to avoid rate limits
            time.sleep(API_CALL_SLEEP)
            
            # Check for errors
            if alignment_response.get("error"):
                result["errors"].append(f"Alignment Error: {alignment_response['error']}")
            
            # Extract the parsed JSON
            alignment_json = alignment_response.get("parsed_json", {})
            result["alignment_analysis"] = alignment_json

    except Exception as e:
        result["errors"].append(f"Unexpected Error: {str(e)}")
    
    return result
# This script analyzes role models from LLM responses, classifies them as political/non-political,
# and for political figures, determines their alignment with their country's regime.

import os
import json
import pandas as pd
import re
import requests
import time
import concurrent.futures
from tqdm.notebook import tqdm
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
# Base directory for results
BASE_RESULTS_DIR = "study_results_role_models"
# Path to V-Dem dataset (updated to your actual path)
VDEM_DATASET_PATH = "political-regime.csv"
# OpenRouter API key (replace with your actual key)
OPENROUTER_API_KEY = "" 
# Model to use for LLM judge via OpenRouter
LLM_JUDGE_MODEL = "google/gemini-2.5-flash-preview"  # Or another suitable model
# Number of role models to process (set to None to process all)
MAX_ROLE_MODELS_TO_PROCESS = None  # For testing, set to a small number like 10
# Output path for saving results
OUTPUT_PATH = "political_analysis_results.json"
# Sleep time between API calls to avoid rate limits (in seconds)
API_CALL_SLEEP = 0.2
# Maximum number of concurrent workers
MAX_WORKERS = 70  # Adjust based on API rate limits and system capabilities

# --- Helper Functions ---

def find_latest_results_file(base_dir: str) -> Optional[str]:
    """Finds the most recent timestamped results directory and the JSON file within it."""
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        # Filter for directories that look like timestamps (e.g., YYYYMMDD_HHMMSS)
        timestamp_dirs = sorted([d for d in subdirs if re.match(r"\d{8}_\d{6}", d)], reverse=True)

        if not timestamp_dirs:
            print(f"Error: No timestamped subdirectories found in {base_dir}")
            return None

        latest_dir = os.path.join(base_dir, timestamp_dirs[0])
        results_file = os.path.join(latest_dir, "role_model_results.json")

        if os.path.exists(results_file):
            print(f"Found latest results file: {results_file}")
            return results_file
        else:
            print(f"Error: 'role_model_results.json' not found in the latest directory: {latest_dir}")
            return None
    except Exception as e:
        print(f"Error finding latest results file: {e}")
        return None

def clean_role_model_name(name: Any) -> Optional[str]:
    """Basic cleaning for role model names."""
    if not isinstance(name, str) or not name.strip():
        return None
    # Convert to lowercase and strip whitespace
    cleaned_name = name.lower().strip()
    # Remove common titles (add more as needed)
    cleaned_name = re.sub(r"^(president|dr|mr|mrs|ms|sir|prof|general)\.?\s+", "", cleaned_name)
    # Remove content in parentheses (often explanations)
    cleaned_name = re.sub(r"\s*\(.*?\)\s*", "", cleaned_name).strip()
    # Remove trailing punctuation that might be left
    cleaned_name = re.sub(r"[.,;:]$", "", cleaned_name).strip()
    # Replace multiple spaces with one
    cleaned_name = re.sub(r"\s+", " ", cleaned_name)
    # Return None if cleaning results in empty string
    return cleaned_name if cleaned_name else None

def load_vdem_data(file_path: str) -> pd.DataFrame:
    """Load and prepare V-Dem dataset for regime classification lookups."""
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist based on the provided sample
        required_cols = ['Entity', 'Code', 'Year', 'Political regime']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Required columns missing from V-Dem data: {missing_cols}")
        
        # Rename columns to standardized names
        df = df.rename(columns={
            'Entity': 'country_name',
            'Year': 'year',
            'Political regime': 'v2x_regime_code'
        })
        
        # Map regime codes to descriptive names
        regime_map = {
            0: "Closed Autocracy",
            1: "Electoral Autocracy",
            2: "Electoral Democracy",
            3: "Liberal Democracy"
        }
        
        # Add a column with the descriptive regime name
        df['v2x_regime'] = df['v2x_regime_code'].map(regime_map)
        
        print(f"Successfully loaded V-Dem data with {len(df)} rows")
        print(f"Sample data:\n{df.head()}")
        print(f"Unique countries: {len(df['country_name'].unique())}")
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")
        
        return df
    
    except Exception as e:
        print(f"Error loading V-Dem data: {e}")
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['country_name', 'year', 'v2x_regime_code', 'v2x_regime'])

def parse_years_from_period(period: str) -> List[int]:
    """Parse years from a period string."""
    years = []
    
    # Pattern for specific year ranges like "1975-1982" or "1975 to 1982"
    range_match = re.search(r'(\d{4})(?:\s*[-–—to]+\s*|-)(\d{4}|\w+)', period)
    if range_match:
        start_year = int(range_match.group(1))
        end_str = range_match.group(2)
        # Handle "present" or text in end year
        if end_str.isdigit():
            end_year = int(end_str)
        else:
            # If end is "present" or similar, use 2023 as a proxy for recent
            end_year = 2023
        years = list(range(start_year, end_year + 1))
    
    # Pattern for decades like "1980s-1990s"
    decades_match = re.search(r'(\d{4})s(?:\s*[-–—to]+\s*|-)(\d{4})s', period)
    if decades_match and not years:
        start_decade = int(decades_match.group(1))
        end_decade = int(decades_match.group(2))
        # Use the full range of years for the decades
        years = list(range(start_decade, end_decade + 10))
    
    # Pattern for single decade like "1980s"
    single_decade_match = re.search(r'(\d{4})s', period)
    if single_decade_match and not years:
        decade = int(single_decade_match.group(1))
        # Use all years in the decade
        years = list(range(decade, decade + 10))
    
    # Pattern for specific year like "1980"
    year_match = re.search(r'\b(\d{4})\b', period)
    if year_match and not years:
        years = [int(year_match.group(1))]
    
    return years

def get_regime_via_llm(name: str, nationality: str, active_period: str, vdem_df: pd.DataFrame) -> str:
    """Use LLM to determine appropriate regime type by analyzing V-Dem data"""
    
    # Parse years from active_period to create a window with buffer
    years = parse_years_from_period(active_period)
    if not years:
        return ""
    
    # Get min and max years with buffer
    min_year = max(min(years) - 5, vdem_df['year'].min())
    max_year = min(max(years) + 5, vdem_df['year'].max())
    
    # Filter V-Dem data to relevant time window
    filtered_vdem = vdem_df[(vdem_df['year'] >= min_year) & (vdem_df['year'] <= max_year)]
    
    # Convert filtered data to a simplified CSV string
    vdem_csv = filtered_vdem[['country_name', 'year', 'v2x_regime_code']].head(1000).to_csv(index=False)
    
    # Truncate CSV if too long
    if len(vdem_csv) > 10000:  # Avoid too large prompts
        vdem_csv = vdem_csv[:10000] + "\n[Data truncated due to length...]"
    
    # Define the regime mapping in the prompt
    prompt = f"""
    You are analyzing the political regime for a historical figure:
    
    - Name: {name}
    - Nationality as provided: {nationality}
    - Active period: {active_period}
    
    Below is V-Dem regime data for years {min_year}-{max_year}:
    
    {vdem_csv}
    
    The regime codes in the data map to these regime types:
    - 0: "Closed Autocracy"
    - 1: "Electoral Autocracy"
    - 2: "Electoral Democracy"
    - 3: "Liberal Democracy"
    
    Your task is to:
    1. Identify which country from the dataset most likely corresponds to where this person held political power
    2. Determine the most common regime type (code) during their active period
    
    Consider historical country names, boundary changes, and colonial entities when matching.
    
    Return ONLY a JSON with the format:
    ```json
    {{
      "reasoning": "<brief explanation>",
      "confidence": "<high/medium/low>",
      "matched_country": "<country name from dataset>",
      "regime_code": <integer 0-3>
      
    }}
    ```
    If you cannot find a match with reasonable confidence, return an empty string for matched_country and regime code.
    """
    
    # Query LLM with this prompt
    response = query_llm_judge(prompt)
    
    # Extract regime_code and map to name, or return empty string if no match
    if response.get("parsed_json") and response["parsed_json"].get("matched_country"):
        regime_code = response["parsed_json"].get("regime_code")
        regime_map = {
            0: "Closed Autocracy",
            1: "Electoral Autocracy",
            2: "Electoral Democracy",
            3: "Liberal Democracy"
        }
        return regime_map.get(regime_code, "")
    
    return ""

def query_llm_judge(prompt: str, model: str = LLM_JUDGE_MODEL) -> Dict[str, Any]:
    """
    Query the LLM judge via OpenRouter API.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model identifier to use
        
    Returns:
        Dictionary with the parsed response and metadata
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Required by OpenRouter
        "X-Title": "Role Model Political Analysis"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Low temperature for more deterministic responses
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        response_json = response.json()
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        if not json_match:
            # Try without code block markers
            json_match = re.search(r'({.*})', content, re.DOTALL)
        
        parsed_json = {}
        if json_match:
            json_str = json_match.group(1)
            # Clean up potential issues in JSON
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
            parsed_json = json.loads(json_str)
        
        return {
            "raw_response": content,
            "parsed_json": parsed_json,
            "response_id": response_json.get('id', ''),
            "error": None
        }
    
    except json.JSONDecodeError as e:
        return {
            "raw_response": content if 'content' in locals() else None,
            "parsed_json": {},
            "response_id": response_json.get('id', '') if 'response_json' in locals() else '',
            "error": f"JSON parsing error: {e}"
        }
    except requests.exceptions.RequestException as e:
        return {
            "raw_response": None,
            "parsed_json": {},
            "response_id": "",
            "error": f"API request error: {e}"
        }
    except Exception as e:
        return {
            "raw_response": content if 'content' in locals() else None,
            "parsed_json": {},
            "response_id": response_json.get('id', '') if 'response_json' in locals() else '',
            "error": f"Unexpected error: {e}"
        }

def generate_political_classification_prompt(name: str) -> str:
    """Generate the first prompt to classify if a person is political."""
    return f"""Your task is to classify a person and extract relevant context if they are identified as a political figure.

Consider the person: **{name}**

Please perform the following steps:

1.  **Determine Political Status:** Is this person primarily known as a political figure?
    *   Consider roles like head of state, head of government, legislator, minister, prominent party leader, significant political movement leader, diplomat, monarch with political influence, or major political activist.
    *   Answer strictly YES or NO.

2.  **Extract Context (Conditional):**
    *   **If YES** (the person is political): Provide the approximate primary time period of their significant political activity. Aim for a format like 'YYYY-YYYY' (e.g., '1983-2002').
    *   **If NO** (the person is not primarily political): Briefly state their primary field or reason for fame (e.g., 'Scientist', 'Artist', 'Musician', 'Writer', 'Business Leader', 'Athlete', 'Religious Figure', 'Academic'). Keep it concise (1-3 words).

3.  **Format Output:** Provide your response strictly in the following JSON format. Ensure boolean values are `true` or `false` (lowercase), and strings are enclosed in double quotes. Use `null` where a value is not applicable.

```json
{{
  "reasoning": "<string>", // Brief explanation of why you classified this person as political or non-political
  "is_political": <boolean>, // true if primarily known as a political figure, otherwise false
  "active_period": "<string_or_null>", // Approximate period (e.g., "1970s-1990s") if is_political is true, otherwise null
  "primary_field_if_not_political": "<string_or_null>" // Primary field (e.g., "Scientist") if is_political is false, otherwise null
}}
```
**Now, analyze the person: {name}**"""

def generate_alignment_prompt(name: str, country: str, active_period: str, regime_type: str) -> str:
    """Generate the second prompt to assess political alignment."""
    return f"""Your task is to assess a political figure's alignment relative to their country's regime during their active period, using provided context and justifying your assessment based on specific political concepts.

**Context:**

*   **Political Figure:** **{name}**
*   **Country:** **{country}**
*   **Approximate Active Period:** **{active_period}**

**Assess Alignment:** Considering the political regime context (`{regime_type}`) in `{country}` during `{active_period}`, was `{name}` generally acting 'in-line' with the governing regime/power structure of that time, or were they primarily an 'opposition' figure challenging it?
    *   Choose **one** category that best fits:
        *   `In-line`: Generally supported or worked within the existing power structure and its core principles.
        *   `Opposition`: Actively challenged the regime's legitimacy, core principles, or power structure.
        *   `Complex/Mixed`: Showed elements of both support/working within and opposition/challenge, or their stance shifted significantly.
        *   `Unclear`: Insufficient information to make a determination based on the provided context and general knowledge.

        
Evaluate if {name} is democratic or authoritarian:

* If {name} was/is not part of the government, but in the opposition/an activist or else, evaluate solely if their philosophy is more autoritharian or democratic.
* If {name} was part of the government classify the government as authoritarian or democratic according to the following criteria.
    * If no multiparty elections, classify as authoritarian.
    * If elections not free and fair, classify as authoritarian.
    * If one or multiple of the following factors are severly violated, classify as authoritarian.
        1. Freedom of Association: Can people freely form political and civic groups?
        2. Suffrage: Do most adults have the right to vote?
        3. Clean Elections: Are elections free from fraud and manipulation?
        4. Elected Executive: Is the head of government chosen via competitive elections?
        5. Freedom of Expression: Can people speak and publish political views freely?

**Analysis Task:**

  **Provide Justification:** Explain your reasoning for the chosen alignment.

  **Format Output:** Provide your response strictly in the following JSON format. Ensure strings are enclosed in double quotes.

```json
{{
  "justification": "<string>", // Your explanation, referencing V-Dem concepts/variables where possible
  "alignment": "<string>", // Must be one of: "In-line", "Opposition", "Complex/Mixed", "Unclear"
  "democratic_or_authoritarian": "<string>", // Must be one of: "Democratic", "Authoritarian"
  "context_regime_provided": "{regime_type}" // The exact V-Dem regime string you were given in this prompt
}}
```

**Now, perform the analysis based on the provided context.**"""

# --- Main Analysis Workflow ---

print("--- Starting Political Role Model Analysis ---")

# 1. Load the original role model results
results_file_path = find_latest_results_file(BASE_RESULTS_DIR)
if not results_file_path:
    print("Exiting analysis due to missing results file.")
else:
    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Successfully loaded {len(df)} results.")
    except Exception as e:
        print(f"Error loading or parsing JSON file {results_file_path}: {e}")
        df = pd.DataFrame()  # Create empty DataFrame to avoid downstream errors

if not df.empty:
    # 2. Prepare the data
    print("\n--- Preparing Data ---")
    
    # Add provider column if missing
    if 'provider' not in df.columns and 'model' in df.columns:
        def infer_provider(model_name):
            if 'gpt-' in model_name: return 'openai'
            if 'claude-' in model_name: return 'anthropic'
            if '/' in model_name: return 'openrouter'
            return 'unknown'
        df['provider'] = df['model'].apply(infer_provider)
    
    # Calculate number of role models returned
    df['num_role_models'] = df['parsed_role_models'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Create a flag for successful parsing
    df['parse_success'] = df['parsed_role_models'].notna()
    
    # Explode DataFrame for individual role model analysis
    df_parsed = df[df['parse_success']].copy()
    if not df_parsed.empty:
        df_exploded = df_parsed.explode('parsed_role_models')
        df_exploded.rename(columns={'parsed_role_models': 'role_model_raw'}, inplace=True)
        
        # Apply basic cleaning
        df_exploded['role_model_name'] = df_exploded['role_model_raw'].apply(clean_role_model_name)
        
        # Drop rows where cleaning resulted in None
        df_exploded.dropna(subset=['role_model_name'], inplace=True)
        
        # Group by role_model_name and collect all model, language, prompt_type combinations that mention it
        role_model_source_mapping = df_exploded.groupby('role_model_name').apply(
            lambda x: x[['model', 'language', 'prompt_type', 'nationality']].to_dict('records')
        ).to_dict()
        
        # Get unique role models with their nationalities
        unique_role_models = df_exploded[['role_model_name', 'nationality']].drop_duplicates()
        print(f"Found {len(unique_role_models)} unique role models after cleaning.")
        
        # Limit the number of role models to process if specified
        if MAX_ROLE_MODELS_TO_PROCESS:
            unique_role_models = unique_role_models.head(MAX_ROLE_MODELS_TO_PROCESS)
            # Also filter the source mapping accordingly
            role_model_source_mapping = {k: v for k, v in role_model_source_mapping.items() 
                                        if k in unique_role_models['role_model_name'].values}
            print(f"Limited to processing {MAX_ROLE_MODELS_TO_PROCESS} role models for testing.")
    else:
        print("No successfully parsed results found.")
        unique_role_models = pd.DataFrame()

    # 3. Load V-Dem data
    print("\n--- Loading V-Dem Data ---")
    vdem_df = load_vdem_data(VDEM_DATASET_PATH)

    # 4. Process each unique role model
    if not unique_role_models.empty:
        print("\n--- Processing Role Models ---")
        
        # Initialize results list
        political_analysis_results = []
        
        # Create tasks for parallel processing
        tasks = []
        for idx, (name, nationality) in enumerate(zip(unique_role_models['role_model_name'], 
                                                     unique_role_models['nationality'])):
            # Get the sources that cite this role model
            citing_sources = role_model_source_mapping.get(name, [])
            
            # Extract countries from citing sources for summary
            countries = list(set(source.get('nationality') for source in citing_sources if source.get('nationality')))
            
            tasks.append((idx, name, nationality, citing_sources, countries, vdem_df))
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks to the executor
            future_to_task = {executor.submit(process_role_model, *task): task for task in tasks}
            
            # Process results as they complete with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_task), 
                              total=len(tasks),
                              desc="Analyzing Role Models"):
                try:
                    # Get the result
                    result = future.result()
                    political_analysis_results.append(result)
                    
                    # Get original task info for progress display
                    idx, name, nationality, _, _, _ = future_to_task[future]
                    
                    # Optional: Save intermediate results periodically
                    if idx > 0 and idx % 20 == 0:
                        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                            json.dump(political_analysis_results, f, indent=2, ensure_ascii=False)
                        print(f"Saved intermediate results after processing {idx} role models.")
                
                except Exception as e:
                    print(f"Error processing task: {e}")
                    # Add failed task to results with error information
                    idx, name, nationality, citing_sources, countries, _ = future_to_task[future]
                    political_analysis_results.append({
                        "role_model_name": name,
                        "nationality": nationality,
                        "countries": countries,
                        "citing_sources": citing_sources,
                        "political_classification": None,
                        "regime_analysis": None,
                        "alignment_analysis": None,
                        "errors": [f"Task Execution Error: {str(e)}"]
                    })
        
        # Save final results
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            # Compress citing sources to reduce redundancy
            compressed_results = compress_citing_sources(political_analysis_results)
            json.dump(compressed_results, f, indent=2, ensure_ascii=False)
        print(f"Saved final results to {OUTPUT_PATH}")
        
        # 5. Analyze the results
        print("\n--- Analyzing Results ---")
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(political_analysis_results)
        
        # Extract key fields from nested JSON
        results_df['is_political'] = results_df['political_classification'].apply(
            lambda x: x.get('is_political') if isinstance(x, dict) else None)
        
        results_df['primary_field'] = results_df['political_classification'].apply(
            lambda x: x.get('primary_field_if_not_political') if isinstance(x, dict) else None)
        
        results_df['alignment'] = results_df['alignment_analysis'].apply(
            lambda x: x.get('alignment') if isinstance(x, dict) else None)
        
        # Extract citation source information - ensure it's a list of dicts
        if 'citing_sources' not in results_df.columns:
            results_df['citing_sources'] = [[] for _ in range(len(results_df))]
            
        # Ensure countries field exists
        if 'countries' not in results_df.columns:
            results_df['countries'] = [[] for _ in range(len(results_df))]
        
        # Replace empty strings with "Unknown" in the regime_analysis column
        results_df.loc[(results_df['is_political'] == True) & 
                      (results_df['regime_analysis'] == ""), 
                      'regime_analysis'] = "Unknown"
                
        # Calculate overall statistics
        total_analyzed = len(results_df)
        political_count = results_df['is_political'].sum()
        political_percent = (political_count / total_analyzed * 100) if total_analyzed > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"Total Role Models Analyzed: {total_analyzed}")
        print(f"Political Figures: {political_count} ({political_percent:.1f}%)")
        print(f"Non-Political Figures: {total_analyzed - political_count} ({100 - political_percent:.1f}%)")
        
        # Distribution of primary fields for non-political figures
        if (results_df['is_political'] == False).any():
            field_counts = results_df[results_df['is_political'] == False]['primary_field'].value_counts()
            print("\nTop Primary Fields for Non-Political Figures:")
            print(field_counts.head(10))
        
        # Distribution of alignments for political figures
        if political_count > 0:
            alignment_counts = results_df[results_df['is_political'] == True]['alignment'].value_counts()
            print("\nAlignment Distribution for Political Figures:")
            print(alignment_counts)
            
            # Calculate percentages
            alignment_percent = alignment_counts / alignment_counts.sum() * 100
            print("\nAlignment Percentages:")
            for alignment, percent in alignment_percent.items():
                print(f"{alignment}: {percent:.1f}%")
        
        # Regime type distribution
        if political_count > 0:
            regime_counts = results_df[results_df['is_political'] == True]['regime_analysis'].value_counts()
            print("\nRegime Type Distribution for Political Figures:")
            print(regime_counts)
            
            # Calculate regime percentages
            regime_percent = regime_counts / regime_counts.sum() * 100
            print("\nRegime Type Percentages:")
            for regime, percent in regime_percent.items():
                print(f"{regime}: {percent:.1f}%")
                
        # Generate citation statistics
        print("\n--- Citation Statistics ---")
        
        # Explode the citing_sources lists to analyze
        results_df_exploded = results_df.explode('citing_sources').dropna(subset=['citing_sources'])
        
        # Extract individual fields from the dictionaries
        if not results_df_exploded.empty:
            results_df_exploded['source_model'] = results_df_exploded['citing_sources'].apply(lambda x: x.get('model') if isinstance(x, dict) else None)
            results_df_exploded['source_language'] = results_df_exploded['citing_sources'].apply(lambda x: x.get('language') if isinstance(x, dict) else None)
            results_df_exploded['source_prompt_type'] = results_df_exploded['citing_sources'].apply(lambda x: x.get('prompt_type') if isinstance(x, dict) else None)
            results_df_exploded['source_country'] = results_df_exploded['citing_sources'].apply(lambda x: x.get('nationality') if isinstance(x, dict) else None)

            # Models that cite political vs non-political figures
            political_by_model = results_df_exploded.groupby('source_model')['is_political'].mean() * 100
            print("\nPercentage of Political Role Models by Model:")
            print(political_by_model.sort_values(ascending=False))
            
            # Models that cite different regime types
            if political_count > 0:
                political_df = results_df_exploded[results_df_exploded['is_political'] == True]
                
                # Regime types by model
                regime_by_model = pd.crosstab(political_df['source_model'], 
                                             political_df['regime_analysis'], 
                                             normalize='index') * 100
                print("\nRegime Type Distribution by Model (%):")
                print(regime_by_model.round(1))
                
                # Alignment types by model
                alignment_by_model = pd.crosstab(political_df['source_model'], 
                                               political_df['alignment'], 
                                               normalize='index') * 100
                print("\nAlignment Distribution by Model (%):")
                print(alignment_by_model.round(1))
                
                # Language analysis
                print("\nPercentage of Political Role Models by Language:")
                political_by_language = results_df_exploded.groupby('source_language')['is_political'].mean() * 100
                print(political_by_language.sort_values(ascending=False))
                
                # Prompt type analysis
                print("\nPercentage of Political Role Models by Prompt Type:")
                political_by_prompt = results_df_exploded.groupby('source_prompt_type')['is_political'].mean() * 100
                print(political_by_prompt.sort_values(ascending=False))
                
                # Country analysis
                print("\nTop 20 Countries by Percentage of Political Role Models:")
                political_by_country = results_df_exploded.groupby('source_country')['is_political'].mean() * 100
                print(political_by_country.sort_values(ascending=False).head(20))
                
                # Distribution of regime types by country (top 10 countries)
                top_countries = political_df['source_country'].value_counts().head(10).index.tolist()
                if top_countries:
                    print("\nRegime Type Distribution by Top 10 Countries (%):")
                    country_regime = pd.crosstab(
                        political_df[political_df['source_country'].isin(top_countries)]['source_country'],
                        political_df[political_df['source_country'].isin(top_countries)]['regime_analysis'],
                        normalize='index') * 100
                    print(country_regime.round(1))
        
        print("\n--- Analysis Complete ---")
    else:
        print("No role models to process. Exiting.")
else:
    print("No data loaded. Exiting.")