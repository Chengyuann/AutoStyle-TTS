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

# Main function
def main(args):
    # Set random seed for reproducibility
    set_random_seed(args.seed)

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

    # Generate embedding for the search text
    try:
        print(f"Generating embedding for the search text: '{args.search_text}'")
        # Generate emotion label
        emotion_label = generate_emotion_label(args.search_text, model, tokenizer, args.device, max_new_tokens=10)
        print(f"Generated emotion label: '{emotion_label}'")

        # Since this is a standalone query script, we don't have a biography.
        # You may need to provide a default biography or retrieve it from your data.
        # For simplicity, we'll use a placeholder biography.
        biography = "This is a placeholder biography based on the input text."
        print(f"Using biography: '{biography}'")

        # Create combined embedding (emotion + biography)
        combined_search_emb = get_embedding(emotion_label, model, tokenizer, args.device, layer=-1, pooling='mean')
        bio_emb = get_embedding(biography, model, tokenizer, args.device, layer=-1, pooling='mean')
        combined_search_emb = np.concatenate((combined_search_emb, bio_emb))  # 3072 + 3072 = 6144
        print(f"Combined embedding shape: {combined_search_emb.shape}")
    except Exception as e:
        print(f"Error generating embedding for the search text: {e}")
        traceback.print_exc()
        return

    # Perform search
    try:
        print(f"Performing search in collection '{collection_name}' with top_k={args.top_k}")
        search_results = search_milvus(client, collection_name, combined_search_emb, top_k=args.top_k, filter_expr=None)

        # Process and display results
        if search_results:
            for query_idx, query_result in enumerate(search_results):
                print(f"\nTop {args.top_k} results for the query '{args.search_text}':")
                for res in query_result:
                    distance = res.get("distance")
                    entity = res.get("entity", {})
                    retrieved_file_id = entity.get("file_id")
                    text = entity.get("text")
                    print(f"File ID: {retrieved_file_id}, Distance: {distance}, Text: {text}")
                print("-" * 50)
        else:
            print("No results found.")
    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc()

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query Milvus Lite with a search text')
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/EHG/BiosERC/finetuned_llm/iemocap_apdcephfs_cq10_ep3_step-1_lrs-linear3e-4_0shot_r32_w5_spdescV2_seed42_L1024_llmdescllama3.2-3b_ED600', 
                        help='Path to the fine-tuned Llama 3.2 model')
    parser.add_argument('--db_path', type=str, default="milvus_demo.db",
                        help='Path to the Milvus Lite database file')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--search_text', type=str, required=True,
                        help='Text to perform search in Milvus')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top similar results to retrieve')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
