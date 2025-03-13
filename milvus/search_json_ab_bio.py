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
from pymilvus import MilvusClient
from tqdm import tqdm
import argparse
import json
import traceback
from collections import defaultdict

# Define set_random_seed function
def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load local fine-tuned Llama model and tokenizer
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

# Search in Milvus Lite
def search_milvus(client, collection_name, embedding, top_k=3, filter_expr=None):
    """
    搜索Milvus集合中与查询向量最相似的top_k个向量。

    Args:
    - client (MilvusClient): Milvus客户端。
    - collection_name (str): 集合名称。
    - embedding (list or numpy.ndarray): 查询向量。
    - top_k (int): 返回的最相似结果数量。
    - filter_expr (str, optional): 过滤表达式。

    Returns:
    - list: 搜索结果列表，每个结果包含 'id', 'distance', 'entity'。
    """
    try:
        results = client.search(
            collection_name=collection_name,
            data=[embedding],  # 确保 data 是一个列表，其中包含一个或多个向量
            limit=top_k,        # 返回的结果数量
            filter=filter_expr, # 过滤表达式
            output_fields=["file_id", "text"]  # 指定要返回的字段
            # 不传递 'anns_field', 'metric_type', 'param'
        )
        return results
    except Exception as e:
        print(f"Error during Milvus search: {e}")
        traceback.print_exc()
        return []

# Function to read input JSON file
def read_input_json(input_json_path):
    """
    读取输入的JSON文件，每行一个JSON对象，包含 'zh_text' 和 'speaker' 字段。

    Args:
    - input_json_path (str): 输入JSON文件路径。

    Returns:
    - list of dict: 读取的JSON对象列表。
    """
    data = []
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    obj = json.loads(line)
                    if 'zh_text' in obj and 'speaker' in obj:
                        data.append(obj)
                    else:
                        print(f"Missing 'zh_text' or 'speaker' in line {line_num}.")
                except json.JSONDecodeError as jde:
                    print(f"JSON decode error in line {line_num}: {jde}")
    except Exception as e:
        print(f"Error reading input JSON file '{input_json_path}': {e}")
        traceback.print_exc()
    return data

# Group data by speaker
def group_by_speaker(data):
    """
    Group data by the 'speaker' field.
    Each speaker has a list of utterances, each containing 'zh_text'.

    Args:
    - data (list of dict): List of JSON objects.

    Returns:
    - dict: 以'speaker'为键，值为该说话人所有'zh_text'的列表。
    """
    speaker_dict = defaultdict(list)
    for sample in data:
        speaker = sample.get('speaker', 'UNKNOWN_SPEAKER')
        text = sample.get('zh_text', '').strip()
        if text:
            speaker_dict[speaker].append(text)
    return speaker_dict

# Main function
# Main function
def main(args):
    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Read input JSON file
    input_data = read_input_json(args.input_json)
    print(f"Loaded {len(input_data)} samples from '{args.input_json}'.")

    if not input_data:
        print("No valid data loaded from input JSON file.")
        return

    # Group data by speaker
    speaker_texts = group_by_speaker(input_data)
    print(f"Found {len(speaker_texts)} unique speakers.")

    if not speaker_texts:
        print("No speakers found in the input data.")
        return

    # Initialize Milvus Lite
    collection_name = "embeddings_biographies_collection"
    dimension = 6144  # 3072 (emotion) + 3072 (biography)
    try:
        client = MilvusClient(args.db_path)
        print(f"Connected to Milvus Lite database at '{args.db_path}'.")
    except Exception as e:
        print(f"Error connecting to Milvus Lite database at '{args.db_path}': {e}")
        traceback.print_exc()
        return

    # Check if the collection exists
    try:
        if not client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' does not exist in the database.")
            return
        else:
            print(f"Collection '{collection_name}' exists and is ready for queries.")
    except Exception as e:
        print(f"Error checking collection existence: {e}")
        traceback.print_exc()
        return

    # Load tokenizer and model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, use_peft=True)
        model.to(args.device)
        model.eval()
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        traceback.print_exc()
        return

    # Generate biography embeddings for each speaker
    speaker_biographies = {}
    for speaker, texts in tqdm(speaker_texts.items(), desc="Generating biographies"):
        full_conversation = "\n".join(texts)
        try:
            biography = generate_biography(model, tokenizer, full_conversation, speaker, args.device, max_new_tokens=args.max_new_tokens)
            speaker_biographies[speaker] = biography
            print(f"Generated biography for speaker '{speaker}'.")
        except Exception as e:
            print(f"Error generating biography for speaker '{speaker}': {e}")
            traceback.print_exc()
            speaker_biographies[speaker] = "This is a placeholder biography."

    # Process each input text and perform search
    results = []
    for sample in tqdm(input_data, desc="Processing and searching texts"):
        zh_text = sample.get('zh_text', '').strip()
        speaker = sample.get('speaker', 'UNKNOWN_SPEAKER')
        if not zh_text:
            print(f"Skipping empty text for speaker '{speaker}'.")
            continue

        # Generate emotion label
        try:
            emotion_label = generate_emotion_label(zh_text, model, tokenizer, args.device, max_new_tokens=10)
            print(f"Emotion label for text: '{emotion_label}'.")
        except Exception as e:
            print(f"Error generating emotion label for text '{zh_text}': {e}")
            traceback.print_exc()
            emotion_label = "neutral"  # Default emotion

        # Retrieve biography for the speaker
        biography = speaker_biographies.get(speaker, "This is a placeholder biography.")

        # ============================
        # 修改部分开始
        # 只使用传记嵌入进行搜索
        # ============================

        try:
            # 生成传记嵌入（3072维）
            bio_emb = get_embedding(biography, model, tokenizer, args.device, layer=-1, pooling='mean')

            # 创建6144维的查询嵌入：前3072维为0，后3072维为传记嵌入
            query_emb = np.concatenate((np.zeros_like(bio_emb), bio_emb))  # [0, ..., 0, bio_emb]

            # 确保查询嵌入的维度为6144
            if query_emb.shape[0] != dimension:
                raise ValueError(f"Query embedding dimension mismatch: {query_emb.shape[0]}, expected={dimension}")

        except Exception as e:
            print(f"Error creating query embedding for speaker '{speaker}': {e}")
            traceback.print_exc()
            continue

        # ============================
        # 修改部分结束
        # ============================

        # Perform search with top_k=1 using query_emb
        try:
            search_results = search_milvus(client, collection_name, query_emb, top_k=1, filter_expr=None)
            if search_results:
                top_result = search_results[0][0]  # Get the top result
                retrieved_file_id = top_result.get("entity", {}).get("file_id", "N/A")
                retrieved_text = top_result.get("entity", {}).get("text", "N/A")
                distance = top_result.get("distance", "N/A")
                # Prefix the file_id with the specified path
                if args.file_prefix_path:
                    full_file_path = os.path.join(args.file_prefix_path, retrieved_file_id)
                else:
                    full_file_path = retrieved_file_id  # No prefix provided

                results.append({
                    "zh_text": zh_text,
                    "speaker": speaker,
                    "retrieved_file_id": full_file_path,
                    "retrieved_text": retrieved_text,
                    "distance": distance
                })
                print(f"Search successful for text: '{zh_text}'. Retrieved file_id: '{full_file_path}', Distance: {distance}.")
            else:
                results.append({
                    "zh_text": zh_text,
                    "speaker": speaker,
                    "retrieved_file_id": "N/A",
                    "retrieved_text": "N/A",
                    "distance": "N/A"
                })
                print(f"No search results found for text: '{zh_text}'.")
        except Exception as e:
            print(f"Error during search for text '{zh_text}': {e}")
            traceback.print_exc()
            results.append({
                "zh_text": zh_text,
                "speaker": speaker,
                "retrieved_file_id": "Error",
                "retrieved_text": "Error",
                "distance": "Error"
            })

    # Save search results to output JSON file
    if args.output_file:
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(args.output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Search results saved to '{args.output_file}'.")
        except Exception as e:
            print(f"Error saving search results to '{args.output_file}': {e}")
            traceback.print_exc()


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings from a JSON file and perform search in Milvus Lite')
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/EHG/BiosERC/finetuned_llm/iemocap_apdcephfs_cq10_ep3_step-1_lrs-linear3e-4_0shot_r32_w5_spdescV2_seed42_L1024_llmdescllama3.2-3b_ED600', 
                        help='Path to the fine-tuned Llama 3.2 model')
    parser.add_argument('--db_path', type=str, default="milvus_demo.db",
                        help='Path to the Milvus Lite database file')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--input_json', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/tts/test_json/test.json',
                        help='Path to the input JSON file containing texts to search')
    parser.add_argument('--output_file', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/tts/test_json/search_results_ab_bio.json',
                        help='Path to save search results as JSON')
    parser.add_argument('--max_new_tokens', type=int, default=250, 
                        help='Maximum number of new tokens to generate for biography and emotion')
    parser.add_argument('--file_prefix_path', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/tts/seg_wav/tonight1/', 
                        help='Path prefix to add before each retrieved file_id')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
