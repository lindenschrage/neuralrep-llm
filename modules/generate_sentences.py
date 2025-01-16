# modules/sentence_generator.py

import os
import json
import time
import certifi
import openai
from openai import OpenAI
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Tuple
import random


from dotenv import load_dotenv
from config import OPENAI_API_KEY, OPENAI_MODEL, CATEGORIES, SENTENCES_DIR, MANIFOLD_SENTENCES_FILE, TEST_SENTENCES_FILE, NUM_SENTENCES_PER_CATEGORY, NUM_TEST_SENTENCES

load_dotenv()

os.environ['SSL_CERT_FILE'] = certifi.where()

client = OpenAI(api_key=OPENAI_API_KEY)

os.makedirs(SENTENCES_DIR, exist_ok=True)

CATEGORY_WORKERS = 4  
API_WORKERS_PER_CATEGORY = 8  
SENTENCES_PER_CATEGORY = NUM_SENTENCES_PER_CATEGORY + NUM_TEST_SENTENCES
MANIFOLD_SENTENCES_COUNT = NUM_SENTENCES_PER_CATEGORY 
TEST_SENTENCES_COUNT = NUM_TEST_SENTENCES
SAVE_INTERVAL = 10  

data_lock = threading.Lock()

def generate_sentences():
    manifold_data = {}
    processed_categories = 0

    # Load existing manifold data if file exists
    if os.path.exists(MANIFOLD_SENTENCES_FILE):
        with open(MANIFOLD_SENTENCES_FILE, 'rb') as f:
            manifold_data = pickle.load(f)
        print(f"Loaded existing manifold data from '{MANIFOLD_SENTENCES_FILE}'.")


    # Identify categories that need processing
    categories_to_process = [cat for cat in CATEGORIES if cat not in manifold_data]
    total_categories = len(categories_to_process)

    if not categories_to_process:
        print("All categories have already been processed. No action needed.")
        return

    print(f"Starting processing of {total_categories} categories...")

    with ThreadPoolExecutor(max_workers=CATEGORY_WORKERS) as executor:
        # Dictionary to keep track of future to category mapping
        future_to_category = {
            executor.submit(generate_sentences_for_category, category): category
            for category in categories_to_process
        }

        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                manifold_sentences = future.result()
                with data_lock:
                    manifold_data[category] = manifold_sentences
                    processed_categories += 1

                    # Save periodically to prevent data loss
                    if processed_categories % SAVE_INTERVAL == 0 or processed_categories == total_categories:
                        save_data(manifold_data)
                        print(f"Progress: {processed_categories}/{total_categories} categories processed and saved.")

            except Exception as e:
                print(f"Error processing category '{category}': {e}")

    # Final save to ensure all data is persisted
    with data_lock:
        save_data(manifold_data)
        print("All categories have been processed and data has been saved.")

def save_data(manifold_data: dict):
    """Saves manifold_data and test_data to their respective files."""
    try:
        with open(MANIFOLD_SENTENCES_FILE, 'wb') as f:
            pickle.dump(manifold_data, f)
    except Exception as e:
        print(f"Failed to save data: {e}")

def generate_sentences_for_category(category: str) -> Tuple[List[str], List[List[str]]]:
    sentences = []
    num_sentences_needed = SENTENCES_PER_CATEGORY
    num_threads = API_WORKERS_PER_CATEGORY

    lock = threading.Lock()  # To ensure thread-safe operations on 'sentences' list

    def api_call(num_sentences_to_request: int):

        prompt = f"""Write {num_sentences_to_request} varied sentences about [category] (either about a specific
                    example of [category] or about [category] in general, but do not use the word [category]).
                Make these normal, simple sentences that can easily classified to be about [category]"""
        
        max_retries = 5
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You generate sentences into a JSON data object."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "sentence_schema",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "sentences": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "description": "Each sentence about the category"
                                        }
                                    }
                                },
                                "required": ["sentences"],
                                "additionalProperties": False
                            }
                        }
                    }
                )

                # Extract the sentences from the response
                text = response.choices[0].message.content.strip()

                # Parse the JSON object
                response_obj = json.loads(text)

                if isinstance(response_obj, dict) and "sentences" in response_obj:
                    generated_sentences = response_obj["sentences"]
                    with lock:
                        sentences.extend(generated_sentences)
                    break  # Successful, exit the retry loop
                else:
                    print(f"Unexpected format for category '{category}'. Response: {text}")
                    break
            except openai.error.RateLimitError as e:
                print(f"Rate limit error for category '{category}': {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached. Skipping this request.")
                    break
            except Exception as e:
                print(f"Error for category '{category}': {e}")
                if 'text' in locals():
                    print("Response text was:", text)
                break

    while len(sentences) < num_sentences_needed:
        sentences_to_get = num_sentences_needed - len(sentences)
        # Ensure at least 10 sentences per thread to avoid too many small requests
        num_threads_current = min(num_threads, max(sentences_to_get // 10, 1))
        num_sentences_per_thread = sentences_to_get // num_threads_current
        remainder_sentences = sentences_to_get % num_threads_current

        with ThreadPoolExecutor(max_workers=num_threads_current) as executor:
            futures = []
            for i in range(num_threads_current):
                n = num_sentences_per_thread + (1 if i < remainder_sentences else 0)
                if n <= 0:
                    continue
                futures.append(executor.submit(api_call, n))

            # Wait for all futures to complete
            for future in as_completed(futures):
                pass  # Results are already added to 'sentences' in api_call

        # Optional: Add a short delay to respect API rate limits
        time.sleep(1)

    # Ensure we have exactly the required number of sentences
    sentences = sentences[:num_sentences_needed]

    # Randomly shuffle and split the sentences into manifold and test
    random.shuffle(sentences)
    manifold_sentences = sentences[:MANIFOLD_SENTENCES_COUNT]

    # Print first 10 sentences for manifold data
    print(f"First 10 manifold sentences for category '{category}':")
    for sentence in manifold_sentences[:10]:
        print(f" - {sentence}")
    print()

    return manifold_sentences

# Entry point for the script
if __name__ == "__main__":

    generate_sentences()
