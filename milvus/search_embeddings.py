# search_embeddings.py

from pymilvus import MilvusClient
import json
import argparse
import os
import traceback

def search_milvus(client, collection_name, embedding, top_k=3):
    """
    Search the Milvus collection for the top_k most similar vectors.
    """
    try:
        param = {"nprobe": 10}
        results = client.search(
            collection_name=collection_name,
            data=[embedding],
            anns_field="vector",
            param=param,
            limit=top_k,
            output_fields=["file_id", "text"]
        )
        return results
    except Exception as e:
        print(f"Error during Milvus search: {e}")
        traceback.print_exc()
        return []

def main(args):
    # Initialize Milvus client
    client = MilvusClient(args.db_path)
    print(f"Connected to Milvus at '{args.db_path}'.")

    # Load query embedding
    try:
        with open(args.query_embedding, 'r', encoding='utf-8') as f:
            query_embedding = json.load(f)
        print(f"Loaded query embedding from '{args.query_embedding}'.")
    except Exception as e:
        print(f"Error loading query embedding: {e}")
        traceback.print_exc()
        return

    # Perform search
    search_results = search_milvus(client, collection_name="embeddings_biographies_collection", embedding=query_embedding, top_k=args.top_k)

    # Display results
    if search_results:
        for i, result in enumerate(search_results):
            print(f"\nTop {args.top_k} results for Query {i+1}:")
            for res in result:
                file_id = res['entity']['file_id']
                text = res['entity']['text']
                distance = res['distance']
                print(f"File ID: {file_id}, Distance: {distance}, Text: {text}")
            print("-" * 50)
    else:
        print("No results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search embeddings in Milvus Lite')
    parser.add_argument('--query_embedding', type=str, required=True, 
                        help='Path to the JSON file containing the query embedding vector')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top similar results to retrieve')
    parser.add_argument('--db_path', type=str, default="milvus_demo.db",
                        help='Path to the Milvus Lite database file')

    args = parser.parse_args()
    main(args)
