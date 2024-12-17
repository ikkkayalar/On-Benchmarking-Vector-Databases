import weaviate
import pickle
import numpy as np
from tqdm import tqdm
import csv
import time
import os

# Load vectors from pickle file
def load_vectors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors


# Load the base and query vectors from pickle files
base_vectors_path = '../../data/sift_af_base_vector.pkl'
query_vectors_path = '../../data/sift_af_query_vector.pkl'
ground_truth_path = '../../data/sift_af_gt_euc.pkl'

base_vectors_with_attributes = load_vectors_from_pickle(base_vectors_path)

#print(base_vectors_with_attributes)

query_vectors_with_attributes = load_vectors_from_pickle(query_vectors_path)

base_vectors = list(base_vectors_with_attributes['vector'])
query_vectors = list(query_vectors_with_attributes['vector'])

print(f"Total vectors in dataset: {len(base_vectors_with_attributes)}")

print('Data loaded.')

# Çıkarılacak attribute bilgilerini belirleyin
attributes = [
    {
        "attr1": bool(attr[0]),
        "attr2": bool(attr[1]),
        "attr3": bool(attr[2])
    }
    for attr in zip(
        base_vectors_with_attributes['attr1'], 
        base_vectors_with_attributes['attr2'], 
        base_vectors_with_attributes['attr3']
    )
]

# Weaviate setup
client = weaviate.Client("http://localhost:8080", startup_period=30)
collection_name = "Trial_Weaviate_euc"
vector_size = 128
batch_size = 1000

# Experiment configuration
distance_metric = "l2-squared"  # Change this for each metric: "l2-squared", "dot", "cosine"
m_values = [8, 16, 32, 64]
ef_construction_values = [64, 128, 256, 512]
ef_search_values = [128, 256, 512]
limit_values = [1, 10, 100]
experiment_types = ['hnsw', 'flat']

# Track completed experiments
results_file = f'results/AF_weaviate_SIFT_results_{distance_metric}.csv'
completed_experiments = set()
if os.path.exists(results_file):
    with open(results_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            completed_experiments.add(tuple(row[:5]))

# Function to insert vectors into Weaviate
def insert_vectors(vectors, ids, attributes):
    start_time = time.time()
    objects = []
    for vector, vector_id, attr in zip(vectors, ids, attributes):
        objects.append({
            "properties": {
                "vector": vector.tolist(),
                "vector_id": str(vector_id),
                "attr1": attr['attr1'],
                "attr2": attr['attr2'],
                "attr3": attr['attr3']
            },
            "vector": vector
        })
    with client.batch(batch_size=batch_size) as batch:
        for obj in objects:
            batch.add_data_object(obj['properties'], class_name=collection_name, vector=obj['vector'])
    end_time = time.time()
    print(f"Inserted {len(vectors)} vectors with boolean attributes into Weaviate")
    return end_time - start_time

def perform_searches(query_vectors_with_attributes, limit, ef_search, index_type):
    result_ids = []
    start_time = time.time()

    for _, elem in query_vectors_with_attributes.iterrows():
        # Define near_vector structure
        near_vector = {"vector": elem["vector"]}
        if index_type == 'hnsw':
            near_vector["ef"] = ef_search

        # Build Weaviate filter for boolean attributes
        weaviate_filter = {
            "operator": "And",
            "operands": [
                {"path": ["attr1"], "operator": "Equal", "valueBoolean": elem["attr1"]},
                {"path": ["attr2"], "operator": "Equal", "valueBoolean": elem["attr2"]},
                {"path": ["attr3"], "operator": "Equal", "valueBoolean": elem["attr3"]}
            ]
        }

        # Perform the search using Weaviate
        search_query = client.query.get(collection_name, ["vector_id"]) \
            .with_near_vector(near_vector) \
            .with_limit(limit) \
            .with_where(weaviate_filter)

        response = search_query.do()
        if response is None or 'data' not in response or 'Get' not in response['data'] or collection_name not in response['data']['Get']:
            print(f"No results found for query: {elem}")
            result_ids.append([])
            continue
        result_ids.append([res['vector_id'] for res in response['data']['Get'][collection_name]])
    
    end_time = time.time()
    return result_ids, end_time - start_time


# Main experiment loop
total_iterations = len(experiment_types) * len(m_values) * len(ef_construction_values) * len(ef_search_values) * len(limit_values)

# Load ground truth
ground_truth = load_vectors_from_pickle(ground_truth_path)

with tqdm(total=total_iterations, desc="Experiment Progress") as pbar:
    # Insert vectors once before all experiments
    if client.schema.exists(collection_name):
        client.schema.delete_class(collection_name)

    schema = {
        "class": collection_name,
        "properties": [
            {
                "name": "vector",
                "dataType": ["number[]"]
            },
            {
                "name": "vector_id",
                "dataType": ["string"]
            },
            {
                "name": "attr1",
                "dataType": ["boolean"]
            },
            {
                "name": "attr2",
                "dataType": ["boolean"]
            },
            {
                "name": "attr3",
                "dataType": ["boolean"]
            }
        ],
        "vectorizer": "none"  # Since we're providing precomputed vectors
    }

    client.schema.create_class(schema)
    vector_ids = list(range(len(base_vectors)))
    inserting_time = insert_vectors(base_vectors, vector_ids, attributes)

    for experiment_type in experiment_types:
        for m in m_values:
            for ef_construct in ef_construction_values:
                for ef_search in ef_search_values:
                    # Update collection schema based on experiment type
                    client.schema.delete_class(collection_name)
                    schema = {
                        "class": collection_name,
                        "properties": [
                            {
                                "name": "vector",
                                "dataType": ["number[]"]
                            },
                            {
                                "name": "vector_id",
                                "dataType": ["string"]
                            },
                            {
                                "name": "attr1",
                                "dataType": ["boolean"]
                            },
                            {
                                "name": "attr2",
                                "dataType": ["boolean"]
                            },
                            {
                                "name": "attr3",
                                "dataType": ["boolean"]
                            }
                        ],
                        "vectorizer": "none"  # Since we're providing precomputed vectors
                    }

                    if experiment_type == 'hnsw':
                        schema["vectorIndexConfig"] = {
                            "skip": False,
                            "distance": distance_metric,
                            "efConstruction": ef_construct,
                            "maxConnections": m,
                            "ef": ef_search
                        }
                    elif experiment_type == 'flat':
                        schema["vectorIndexType"] = "flat"
                        schema["vectorIndexConfig"] = {
                            "distance": distance_metric
                        }

                    client.schema.create_class(schema)

                    # Reinsert vectors after schema update
                    vector_ids = list(range(len(base_vectors)))
                    inserting_and_indexing_time = insert_vectors(base_vectors, vector_ids, attributes)

                    for limit in limit_values:
                        experiment_key = (experiment_type, m, ef_construct, ef_search, limit)
                        if experiment_key in completed_experiments:
                            pbar.update(1)
                            continue

                        result_ids, search_time = perform_searches(query_vectors_with_attributes, limit, ef_search, experiment_type)
                        # Calculate recall
                        true_positives = 0
                        n_classified = 0
                        for i, elem in enumerate(result_ids):
                            true_positives_iter = len(np.intersect1d(ground_truth[i][:limit], result_ids[i]))
                            true_positives += true_positives_iter
                            n_classified += len(elem)
                        recall = true_positives / n_classified

                        # Calculate latency and throughput
                        avg_latency = search_time / len(query_vectors)
                        qps = len(query_vectors) / search_time

                        # Log results
                        with open(results_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            if file.tell() == 0:  # Write header only if file is empty
                                writer.writerow(['Experiment Type', 'M', 'Ef Construct', 'Ef Search', 'Limit', 'Avg Latency', 'Throughput (QPS)', 'Recall', 'Inserting Time', 'Inserting and Indexing Time', 'Search Time'])
                            writer.writerow([experiment_type, m, ef_construct, ef_search, limit, avg_latency, qps, recall, inserting_time, inserting_and_indexing_time, search_time])

                        pbar.update(1)

print(f"Experiment for {distance_metric} completed and results saved to {results_file}.")
