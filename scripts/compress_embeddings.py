import argparse
import json
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def generate_dummy_data(output_path, num_nodes=100, dim=128):
    """Generate dummy high-dimensional embeddings for testing."""
    data = {}
    for i in range(num_nodes):
        data[f"node_{i}"] = np.random.randn(dim).tolist()
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Generated dummy data with {num_nodes} nodes of dimension {dim} at {output_path}")

def compress_embeddings(input_file, output_file, target_dim=8):
    """
    Compresses high-dimensional node embeddings into a compact format
    (e.g., 8-dimensional uint8 arrays) suitable for smart contract gating.
    """
    # 1. Load embeddings
    # Assuming input is a JSON file mapping node_id -> [float, float, ...]
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Generating dummy data...")
        generate_dummy_data(input_file)

    with open(input_file, 'r') as f:
        data = json.load(f)
        
    node_ids = list(data.keys())
    embeddings = np.array(list(data.values()))
    
    print(f"Loaded {len(node_ids)} embeddings of dimension {embeddings.shape[1]}")
    
    # 2. Dimensionality reduction using PCA
    if embeddings.shape[1] > target_dim:
        print(f"Reducing dimensionality to {target_dim} using PCA...")
        pca = PCA(n_components=target_dim)
        reduced_embeddings = pca.fit_transform(embeddings)
        variance_retained = sum(pca.explained_variance_ratio_)
        print(f"Variance retained: {variance_retained:.2%}")
    else:
        reduced_embeddings = embeddings
        
    # 3. Quantization to uint8 (0-255)
    print("Quantizing embeddings to uint8...")
    scaler = MinMaxScaler(feature_range=(0, 255))
    quantized_embeddings = scaler.fit_transform(reduced_embeddings).astype(np.uint8)
    
    # 4. Format for smart contract (hex strings or lists of ints)
    contract_ready_data = {}
    for i, node_id in enumerate(node_ids):
        # We can store as a list of integers or a hex string
        # A hex string is often easiest to pass as bytes/bytearray to a smart contract
        hex_string = quantized_embeddings[i].tobytes().hex()
        contract_ready_data[node_id] = {
            "values": quantized_embeddings[i].tolist(),
            "hex": f"0x{hex_string}"
        }
        
    # 5. Save output
    with open(output_file, 'w') as f:
        json.dump(contract_ready_data, f, indent=2)
        
    print(f"Successfully compressed embeddings and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress node embeddings for smart contract gating.")
    parser.add_argument("--input", default="embeddings.json", help="Path to input JSON file (node_id -> float array)")
    parser.add_argument("--output", default="compressed_embeddings.json", help="Path to output JSON file")
    parser.add_argument("--dim", type=int, default=8, help="Target dimensionality (default: 8)")
    
    args = parser.parse_args()
    compress_embeddings(args.input, args.output, args.dim)
