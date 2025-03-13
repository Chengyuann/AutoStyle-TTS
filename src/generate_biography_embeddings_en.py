# Install necessary libraries (if not already installed)
# pip install -U pymilvus transformers torch peft sentencepiece tqdm

# Import necessary libraries
import os
import torch
import numpy as np
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM  # If using PEFT
from transformers import LlamaForCausalLM  # If not using PEFT
from transformers import BitsAndBytesConfig
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import argparse
import json
import random
import traceback
from glob import glob

# Define set_random_seed function
def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_milvus(collection_name, dimension, db_path="milvus_demo.db"):
    """
    Initialize Milvus Lite and create a collection using the simplest setup.
    
    Args:
    - collection_name (str): Name of the collection.
    - dimension (int): Dimension of the vectors.
    - db_path (str): Path to the Milvus Lite database file.
    
    Returns:
    - MilvusClient: The initialized Milvus client.
    """
    try:
        # 创建 MilvusClient 实例
        client = MilvusClient(db_path)
    
        # 如果集合已存在，则删除
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
            print(f"Existing collection '{collection_name}' dropped.")
    
        # 创建集合，仅传递 collection_name 和 dimension
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension
        )
        print(f"Collection '{collection_name}' created successfully with quick setup.")
    
    except Exception as e:
        print(f"Error creating collection '{collection_name}': {e}")
        traceback.print_exc()
    
    return client

# Load local fine-tuned Llama 3.2 model and tokenizer
def load_model_and_tokenizer(model_path, use_peft=True):
    """
    Load the fine-tuned Llama model and tokenizer.

    Args:
    - model_path (str): Path to the fine-tuned model.
    - use_peft (bool): Whether to use PEFT.

    Returns:
    - model: Loaded model.
    - tokenizer: Loaded tokenizer.
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
        # Remove unused keyword arguments
    )

    # Load fine-tuned model (choose based on PEFT usage)
    if use_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Use float16
            quantization_config=quantization_config
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Use float16
            quantization_config=quantization_config
        )

    model.eval()  # Set model to evaluation mode
    return model, tokenizer

# Define embedding extraction function
def get_embedding(text, model, tokenizer, device, layer=-1, pooling='mean'):
    """
    Extract embedding for given text.

    Args:
    - text (str): Input text.
    - model: Fine-tuned Llama model.
    - tokenizer: Corresponding tokenizer.
    - device (torch.device): Device to run the model on.
    - layer (int): Hidden layer index to extract. Default is last layer.
    - pooling (str): Pooling method, supports 'mean' and 'cls'.

    Returns:
    - numpy.ndarray: Embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True
        )
    hidden_states = outputs.hidden_states[layer]  # Get specified hidden layer

    # Convert to float32 and then to NumPy array
    if pooling == 'mean':
        sentence_embedding = hidden_states.mean(dim=1).squeeze().cpu().float().numpy()
    elif pooling == 'cls':
        # Llama model does not have CLS token, recommend using mean pooling
        sentence_embedding = hidden_states[:, 0, :].squeeze().cpu().float().numpy()
    else:
        raise ValueError("Unsupported pooling method")

    return sentence_embedding

# Function to generate biography based on conversation
def generate_biography(model, tokenizer, full_conversation, speaker_name, device, max_new_tokens=250):
    """
    Generate biography for a speaker based on their full conversation.

    Args:
    - model: Loaded language model.
    - tokenizer: Corresponding tokenizer.
    - full_conversation (str): The full conversation of the speaker.
    - speaker_name (str): Name of the speaker.
    - device (torch.device): Device to run the model on.
    - max_new_tokens (int): Maximum tokens to generate.

    Returns:
    - str: Generated biography text.
    """
    # Define biography generation template
    prompting = f"""
Given this conversation between speakers: 
"
{full_conversation}
"
In overall of above conversation, what do you think about the characteristics of speaker {speaker_name}? (Note: provide an answer within 250 words)
"""

    inputs = tokenizer(prompting, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,      # Enable sampling for diversity
            temperature=0.7,     # Adjust temperature for randomness
            top_p=0.9,           # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    biography = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the generated text
    biography = biography.replace(prompting, '').strip()
    return biography

# Function to generate emotion label for a specific utterance
def generate_emotion_label(text, model, tokenizer, device, max_new_tokens=10):
    """
    Generate emotion label for a given utterance.

    Args:
    - text (str): The utterance text.
    - model: Loaded language model.
    - tokenizer: Corresponding tokenizer.
    - device (torch.device): Device to run the model on.
    - max_new_tokens (int): Maximum tokens to generate.

    Returns:
    - str: Emotion label.
    """
    few_shot_example = """\n=======
Context: Given predefined emotional label set [happy, sad, neutral, angry, excited, frustrated], and below conversation: 
"
{}
"

Question: What is the emotion of the speaker at the utterance "{}"?
Answer:"""

    # Fill in the template with conversation and specific utterance
    prompt = few_shot_example.format(text, text)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # Disable sampling for consistency
            temperature=0.0,      # No randomness
            top_p=1.0,            # No nucleus sampling
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    # Decode the generated emotion
    emotion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    emotion = emotion.strip().lower()

    # Optionally, map emotion to standardized labels if necessary
    # For simplicity, assume the model outputs one of the predefined labels

    return emotion

# Function to create combined embedding (emotion + biography)
def create_combined_embedding(emotion_text, biography_text, model, tokenizer, device, dimension=6144):
    """
    Create concatenated embedding vector from emotion and biography texts.

    Args:
    - emotion_text (str): The text for emotion embedding.
    - biography_text (str): The biography text.
    - model: Loaded language model.
    - tokenizer: Corresponding tokenizer.
    - device (torch.device): Device to run the model on.
    - dimension (int): Expected dimension of the combined embedding.

    Returns:
    - numpy.ndarray: Combined embedding vector.
    """
    # Generate emotion embedding
    emotion_emb = get_embedding(emotion_text, model, tokenizer, device, layer=-1, pooling='mean')

    # Generate biography embedding
    bio_emb = get_embedding(biography_text, model, tokenizer, device, layer=-1, pooling='mean')

    # Ensure both embeddings have the same dimension (e.g., 3072)
    if emotion_emb.shape[0] != (dimension // 2) or bio_emb.shape[0] != (dimension // 2):
        raise ValueError(f"Embedding dimension mismatch: emotion_emb={emotion_emb.shape[0]}, bio_emb={bio_emb.shape[0]}, expected={dimension // 2}")

    # Concatenate embeddings
    combined_emb = np.concatenate((emotion_emb, bio_emb))  # 3072 + 3072 = 6144

    return combined_emb

# Load all JSON files
def load_all_json_files(input_paths):
    """
    Load all JSON files from the specified paths.
    If a path is a directory, load all .json files within it.
    If a path is a file, load that file directly.
    Supports standard JSON and JSON Lines (JSONL) formats.
    """
    data = []
    json_files = []
    for input_path in input_paths:
        if os.path.isfile(input_path):
            json_files.append(input_path)
        elif os.path.isdir(input_path):
            json_files.extend(glob(os.path.join(input_path, '*.json')))
        else:
            print(f"Invalid data path: {input_path}")

    for file_path in json_files:
        loaded_samples = 0
        skipped_samples = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)  # Reset file pointer
                if first_char == '[':
                    # Standard JSON list
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        for item in file_data:
                            if isinstance(item, dict):
                                if 'speaker' in item and 'zh_text' in item and 'file_id' in item:
                                    data.append(item)
                                    loaded_samples += 1
                                else:
                                    skipped_samples += 1
                            else:
                                print(f"Unsupported item format in file {file_path}.")
                    else:
                        print(f"Unsupported JSON structure in file {file_path}.")
                else:
                    # Assume JSONL format
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                if 'speaker' in obj and 'zh_text' in obj and 'file_id' in obj:
                                    data.append(obj)
                                    loaded_samples += 1
                                else:
                                    skipped_samples += 1
                            else:
                                print(f"Unsupported object format in file {file_path} at line {line_num}.")
                        except json.JSONDecodeError as jde:
                            print(f"JSON decode error in file {file_path} at line {line_num}: {jde}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            traceback.print_exc()
            continue
        print(f"Loaded {loaded_samples} samples from '{file_path}'. Skipped {skipped_samples} samples due to missing fields.")
    return data

# Group data by speaker
def group_by_speaker(data):
    """
    Group data by the 'speaker' field.
    Each speaker has a list of utterances, each containing 'file_id', 'text', 'emotion'.
    """
    speaker_dict = {}
    for sample in data:
        speaker = sample.get('speaker', 'UNKNOWN_SPEAKER')
        text = sample.get('zh_text', '').strip()
        emotion = sample.get('emotion', 'neutral')  # Default emotion is 'neutral'
        file_id = sample.get('file_id', None)

        if not text or not file_id:
            continue  # Skip samples without text or file_id

        if speaker not in speaker_dict:
            speaker_dict[speaker] = []
        speaker_dict[speaker].append({
            'file_id': file_id,
            'text': text,
            'emotion': emotion
        })
    return speaker_dict

# Extract base filename without extension
def extract_base_filename(file_path):
    """
    Extract the base name of the file without extension.
    For example, '/path/to/Emma_conan.json' -> 'Emma_conan'
    """
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name
# Search in Milvus Lite
def search_milvus(client, collection_name, embedding, top_k=3):
    """
    搜索Milvus集合中与查询向量最相似的top_k个向量。

    Args:
    - client (MilvusClient): Milvus客户端。
    - collection_name (str): 集合名称。
    - embedding (list or numpy.ndarray): 查询向量。
    - top_k (int): 返回的最相似结果数量。

    Returns:
    - list: 搜索结果列表，每个结果包含 'file_id' 和 'distance'。
    """
    try:
        results = client.search(
            collection_name=collection_name,
            data=[embedding],  # 确保 data 是一个列表，其中包含一个或多个向量
            limit=top_k,        # 返回的结果数量
            output_fields=["file_id", "text"]  # 指定要返回的字段
            # 不传递 'anns_field', 'metric_type', 'param'
        )
        return results
    except Exception as e:
        print(f"Error during Milvus search: {e}")
        traceback.print_exc()
        return []



# Main function
def main(args):
    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Load all JSON files
    data = load_all_json_files(args.data_folder)
    print(f"Loaded {len(data)} samples from '{args.data_folder}'.")

    if not data:
        print("No data loaded. Please check the data format and path.")
        return

    # Group data by speaker
    speaker_dict = group_by_speaker(data)
    print(f"Found {len(speaker_dict)} unique speakers.")

    if not speaker_dict:
        print("No speakers found. Please check the data contents.")
        return

    # Load tokenizer and model
    model, tokenizer = load_model_and_tokenizer(args.model_path, use_peft=True)
    model.to(args.device)
    model.eval()

    # Dynamically generate output file name
    if len(args.data_folder) == 1 and os.path.isfile(args.data_folder[0]):
        input_filename = extract_base_filename(args.data_folder[0])
    else:
        input_filename = "multiple_files"

    # Automatically generate output_file if not provided
    if not args.output_file:
        args.output_file = f"embeddings_biographies_en_{input_filename}.json"
    else:
        # If output_file is a directory, generate a file name inside it
        if os.path.isdir(args.output_file):
            output_dir = args.output_file
            output_base = f"embeddings_biographies_en_{input_filename}.json"
            args.output_file = os.path.join(output_dir, output_base)
        elif len(args.data_folder) > 1:
            # Multiple input files, append 'multiple_files'
            output_dir = os.path.dirname(args.output_file)
            output_base = os.path.splitext(os.path.basename(args.output_file))[0]
            new_output_filename = f"{output_base}_multiple_files.json"
            args.output_file = os.path.join(output_dir, new_output_filename)
        elif len(args.data_folder) == 1 and os.path.isfile(args.data_folder[0]):
            # Single input file, append its base name
            base_input = extract_base_filename(args.data_folder[0])
            output_dir = os.path.dirname(args.output_file)
            output_base = os.path.splitext(os.path.basename(args.output_file))[0]
            new_output_filename = f"{output_base}_{base_input}.json"
            args.output_file = os.path.join(output_dir, new_output_filename)

    print(f"Results will be saved to '{args.output_file}'.")

    # Initialize Milvus Lite
    collection_name = "embeddings_biographies_collection"
    dimension = 3072 * 2  # 3072 (emotion) + 3072 (biography)
    try:
        client = initialize_milvus(collection_name, dimension, db_path=args.db_path)
    except Exception as e:
        print(f"Error initializing Milvus: {e}")
        traceback.print_exc()
        return

    # Initialize lists to store results and Milvus insert data
    results = []
    milvus_insert_data = []

    # Process each speaker
    for speaker, utterances in tqdm(speaker_dict.items(), desc="Processing speakers"):
        sentences = [u['text'] for u in utterances]
        emotions = [u['emotion'] for u in utterances]
        file_ids = [u['file_id'] for u in utterances]

        if not sentences:
            continue  # Skip speakers without sentences

        try:
            # Aggregate full conversation for the speaker
            full_conversation = "\n".join([u['text'] for u in utterances])

            # Generate biography for the speaker based on full conversation
            biography = generate_biography(model, tokenizer, full_conversation, speaker, args.device, max_new_tokens=args.max_new_tokens)

            # Generate biography embedding
            bio_emb = get_embedding(biography, model, tokenizer, args.device, layer=-1, pooling='mean')
            bio_emb = bio_emb.astype(np.float32)  # Ensure data type is float32

            for i, (file_id, text, emotion_label) in enumerate(zip(file_ids, sentences, emotions)):
                # Generate emotion label using the provided template
                emotion = generate_emotion_label(text, model, tokenizer, args.device, max_new_tokens=10)
                
                # Create combined embedding (emotion + biography)
                combined_emb = create_combined_embedding(emotion, biography, model, tokenizer, args.device, dimension=dimension)
                
                # Ensure combined embedding has correct dimension
                if combined_emb.shape[0] != dimension:
                    print(f"Embedding dimension mismatch: {combined_emb.shape[0]} != {dimension}")
                    continue

                # Prepare data for Milvus
                milvus_insert_data.append({
                    "file_id": file_id,  # VARCHAR as dynamic field
                    "vector": combined_emb.tolist(),
                    "text": text         # VARCHAR as dynamic field
                })

                # Record result
                result = {
                    'file_id': file_id,
                    'speaker': speaker,
                    'text': text,
                    'emotion': emotion,
                    'biography': biography,
                    'combined_embedding_shape': list(combined_emb.shape)
                }
                results.append(result)

                # Print information
                print(f"File ID: {file_id}")
                print(f"Speaker: {speaker}")
                print(f"Text: {text}")
                print(f"Emotion: {emotion}")
                print(f"Biography: {biography}")
                print(f"Combined Embedding Shape: {combined_emb.shape}")
                print("="*50)

        except Exception as e:
            print(f"Error processing speaker {speaker}: {e}")
            traceback.print_exc()

    # Insert data into Milvus Lite
    if milvus_insert_data:
        # Prepare data for insertion
        insert_data = [
            {
                "id": i,  
                "file_id": item["file_id"],  # VARCHAR as dynamic field
                "vector": item["vector"],    # FLOAT_VECTOR
                "text": item["text"]         # VARCHAR as dynamic field
            }
            for i, item in enumerate(milvus_insert_data, start=1) 
        ]

        # Insert data into Milvus
        try:
            client.insert(collection_name=collection_name, data=insert_data)
            print("Combined embeddings inserted successfully.")
        except Exception as e:
            print(f"Error inserting data into Milvus: {e}")
            traceback.print_exc()
    else:
        print("No data to insert into Milvus.")

    # Optional: Save results to JSON file
    if args.output_file:
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Embeddings and biographies saved to '{args.output_file}'.")
        except Exception as e:
            print(f"Error saving to '{args.output_file}': {e}")
            traceback.print_exc()

    # Optional: Verify inserted data
    if milvus_insert_data:
        print("\nVerifying inserted embeddings by searching with inserted vectors:")
        for i, item in enumerate(milvus_insert_data):
            query_vector = item["vector"]
            search_results = search_milvus(client, collection_name, query_vector, top_k=1)
            if search_results:
                for res in search_results[0]:
                    retrieved_id = res.get('id')   
                    distance = res.get('distance')   
                    print(f"Query ID: {i+1}, Retrieved ID: {retrieved_id}, Distance: {distance}")
            else:
                print(f"No results found for Query ID: {i}")

    # 新增：针对问题一，添加搜索功能
    if args.search_text:
        print(f"\nPerforming search for the input text: '{args.search_text}'")
        try:
            # 生成情感标签
            emotion_label = generate_emotion_label(args.search_text, model, tokenizer, args.device, max_new_tokens=10)

            # 选择一个说话者的传记，或生成一个默认传记
            if speaker_dict:
                first_speaker = list(speaker_dict.keys())[0]
                full_conversation = "\n".join([u['text'] for u in speaker_dict[first_speaker]])
                biography = generate_biography(model, tokenizer, full_conversation, first_speaker, args.device, max_new_tokens=args.max_new_tokens)
            else:
                biography = "unknown"

            # 创建组合嵌入向量
            combined_search_emb = create_combined_embedding(emotion_label, biography, model, tokenizer, args.device, dimension=dimension)

            # Perform search
            search_results = search_milvus(client, collection_name, combined_search_emb, top_k=args.top_k)

            # Process and display results
            if search_results:
                for query_idx, query_result in enumerate(search_results):
                    print(f"\nTop {args.top_k} results for the query '{args.search_text}':")
                    for res in query_result:
                        retrieved_file_id = res.get("file_id") 
                        distance = res.get("distance")   
                        text = res.get("text")    
                        print(f"File ID: {retrieved_file_id}, Distance: {distance}, Text: {text}")
                    print("-" * 50)
            else:
                print("No results found.")

        except Exception as e:
            print(f"Error during search: {e}")
            traceback.print_exc()

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate emotion and biography embeddings and store them in Milvus Lite')
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/EHG/BiosERC/finetuned_llm/iemocap_apdcephfs_cq10_ep3_step-1_lrs-linear3e-4_0shot_r32_w5_spdescV2_seed42_L1024_llmdescllama3.2-3b_ED600', 
                        help='Path to the fine-tuned Llama 3.2 model')
    parser.add_argument('--data_folder', type=str, nargs='+', default=[
        '/apdcephfs_cq10/share_1615176/cq2/macy/tts/json_talk/Tonight1.json', 
        '/apdcephfs_cq10/share_1615176/cq2/rodenluo/tts_vc/test_data/style/emma_conan/Emma_conan.json'
    ], 
                        help='Path to folder containing JSON data files or multiple JSON file paths')
    parser.add_argument('--output_file', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/EHG/BiosERC/output_emb.json', 
                        help='Path to save embeddings and biographies as JSON. If a directory is provided, the script will generate a file name automatically based on input file names.')
    parser.add_argument('--db_path', type=str, default="milvus_demo.db",
                        help='Path to the Milvus Lite database file')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--max_new_tokens', type=int, default=250, 
                        help='Maximum number of new tokens to generate for biography and emotion')
    parser.add_argument('--search_text', type=str, default="A man with a humorous style, from the countryside, with a very delicate mind",
                        help='Text to perform search in Milvus')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top similar results to retrieve')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
