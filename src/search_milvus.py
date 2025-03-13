# Install necessary libraries (if not already installed)
# pip install -U pymilvus transformers torch peft sentencepiece tqdm

import os
import torch
import numpy as np
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM  # If using PEFT
from transformers import LlamaForCausalLM  # If not using PEFT
from transformers import BitsAndBytesConfig
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from tqdm import tqdm
import argparse
import json
import random
import traceback

# Define set_random_seed function
def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    - device: Device to run the model on.
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

# Define speaker biography data
speaker_bio = {
    "ELIZABETH": "Elizabeth is a deeply emotional and passionate individual, extremely devoted to her husband Larry, willing to risk her life to hold onto their relationship.",
    "JOHN": "John is a pragmatic and thoughtful person, always considering the practical aspects of any situation. He values honesty and reliability in his relationships.",
    "PATRICIA": "Patricia is a thoughtful and reserved individual who values her privacy and independence. Her interactions reveal a sense of determination and a desire for meaningful connections.",
    "ANN": "Ann is a cheerful and optimistic person, always looking for the silver lining in every situation. She enjoys connecting with others and values strong interpersonal relationships."
}

def emb_text_bio(speaker, model, tokenizer, device):
    """
    Generate speaker biography embedding.
    """
    bio_text = speaker_bio.get(speaker.upper(), "unknown")
    return get_embedding(bio_text, model, tokenizer, device, layer=-1, pooling='mean')

# Search in Milvus Lite
def search_milvus(client, collection_name, embedding, top_k=3):
    """
    Search the Milvus collection for the top_k most similar vectors.

    Args:
    - client (MilvusClient): Initialized Milvus client.
    - collection_name (str): Name of the collection.
    - embedding (list or numpy.ndarray): Query vector.
    - top_k (int): Number of top similar results to return.

    Returns:
    - list: List of search results with 'file_id' and 'distance'.
    """
    try:
        results = client.search(
            collection_name=collection_name,
            data=[embedding],
            anns_field="vector",
            metric_type="COSINE",
            limit=top_k,
            output_fields=["file_id"]  # Retrieve the 'file_id' field
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

    # Load tokenizer and model
    model, tokenizer = load_model_and_tokenizer(args.model_path, use_peft=True)
    model.to(args.device)
    model.eval()

    # Connect to Milvus
    try:
        client = MilvusClient(args.db_path)
        print(f"Connected to Milvus database at '{args.db_path}'.")
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        traceback.print_exc()
        return

    # Collection name
    collection_name = args.collection_name

    # Verify collection exists
    if not client.has_collection(collection_name=collection_name):
        print(f"Collection '{collection_name}' does not exist. Please check the collection name.")
        return

    # Retrieve and print collection info
    try:
        collection_info = client.get_collection_info(collection_name)
        print(f"Collection '{collection_name}' info:")
        print(collection_info)
    except Exception as e:
        print(f"Error retrieving collection info: {e}")
        traceback.print_exc()
        return

    # Generate embedding for the search query
    query_text = args.search_text
    try:
        # 使用与插入数据相同的方法生成嵌入
        # 如果在插入数据时使用了特殊的嵌入方法，请确保在搜索时也使用相同的方法
        # 在此示例中，我们假设使用与插入时相同的模型和方法
        # 如果您的嵌入方法不同，请相应调整
        # 注意：确保生成的嵌入维度与集合中的向量维度一致
        # 在您的案例中，集合的维度为6144
        # 确保生成的嵌入也是6144维
        # 例如，您可能是通过拼接情感嵌入和传记嵌入得到6144维

        # 示例：假设查询文本对应的 speaker 是已知的，例如 "ELIZABETH"
        # 您可以根据实际情况调整
        # 如果查询文本对应多个 speaker，请相应调整

        # 在此示例中，我们假设查询文本对应于 "ELIZABETH"
        # 并生成情感嵌入和传记嵌入后拼接

        # 定义要查询的 speaker
        query_speaker = args.query_speaker

        # 生成情感嵌入
        emotion_emb = get_embedding(query_text, model, tokenizer, args.device, layer=-1, pooling='mean')

        # 生成传记嵌入
        bio_emb = emb_text_bio(query_speaker, model, tokenizer, args.device)

        # 拼接嵌入
        combined_emb = np.concatenate((emotion_emb, bio_emb))
        combined_emb = combined_emb.astype(np.float32).tolist()

        print(f"Generated combined embedding of shape {len(combined_emb)}.")
    except Exception as e:
        print(f"Error generating embedding for the query text: {e}")
        traceback.print_exc()
        return

    # Perform search
    search_results = search_milvus(client, collection_name, combined_emb, top_k=args.top_k)

    # Process and display results
    if search_results:
        for query_idx, query_result in enumerate(search_results):
            print(f"\nTop {args.top_k} results for the query '{query_text}':")
            for res in query_result:
                retrieved_file_id = res['entity']['file_id']
                distance = res['distance']
                print(f"File ID: {retrieved_file_id}, Distance: {distance}")
            print("-" * 50)
    else:
        print("No results found.")

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search embeddings in Milvus Lite')

    parser.add_argument('--db_path', type=str, default="milvus_demo.db",
                        help='Path to the Milvus Lite database file')
    parser.add_argument('--collection_name', type=str, default="embeddings_biographies_collection",
                        help='Name of the Milvus collection to search')
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1615176/cq2/macy/EHG/BiosERC/finetuned_llm/iemocap_apdcephfs_cq10_ep3_step-1_lrs-linear3e-4_0shot_r32_w5_spdescV2_seed42_L1024_llmdescllama3.2-3b_ED600',
                        help='Path to the fine-tuned Llama 3.2 model')
    parser.add_argument('--search_text', type=str, default="A man with a humorous style, from the countryside, with a very delicate mind",
                        help='Text to perform search in Milvus')
    parser.add_argument('--query_speaker', type=str, default="ELIZABETH",
                        help='Speaker associated with the search text')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top similar results to retrieve')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
