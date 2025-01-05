import weaviate
import pickle
import numpy as np
from tqdm import tqdm
import csv
import time
import os
from utils import fvecs_read

# Function to load vectors from a pickle file
def load_vectors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors

# Load the base and query vectors

base_vectors = list(load_vectors_from_pickle('../../data/base_vectors_gist.pkl'))
query_vectors = list(load_vectors_from_pickle('../../data/query_vectors_gist.pkl'))

ground_truth_path = '../../data/ground_truth_cosine_gist.pkl'

print('Data loaded.')

# Weaviate setup
client = weaviate.Client("http://localhost:8080", startup_period=30)
collection_name = "Trial"
vector_size = 960
batch_size = 1000

# Experiment configuration
distance_metric = "cosine"  # Change this for each metric: "l2-squared", "dot", "cosine"
m_values = [8, 16, 32, 64]
ef_construction_values = [64, 128, 256, 512]
ef_search_values = [128, 256, 512]
limit_values = [1, 10, 100]
experiment_types = ['hnsw', 'flat']

# Track completed experiments
results_file = f'results/ANN_weaviate_GIST_results_{distance_metric}.csv'
completed_experiments = set()
if os.path.exists(results_file):
    with open(results_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            completed_experiments.add(tuple(row[:5]))

# Function to insert vectors into Weaviate
def insert_vectors(vectors, ids):
    start_time = time.time()
    objects = []
    for vector, vector_id in zip(vectors, ids):
        objects.append({
            "properties": {
                "vector": vector.tolist(),
                "vector_id": str(vector_id)
            },
            "vector": vector
        })
    with client.batch(batch_size=batch_size) as batch:
        for obj in objects:
            batch.add_data_object(obj['properties'], class_name=collection_name, vector=obj['vector'])
    end_time = time.time()
    print(f"Inserted {len(vectors)} vectors into Weaviate")
    return end_time - start_time

# Define search function for Weaviate
def perform_searches(query_vectors, limit, ef_search, index_type):
    result_ids = []
    start_time = time.time()
    for query in query_vectors:
        # Define near_vector structure and add ef parameter if HNSW index is used
        near_vector = {"vector": query.tolist()}
        
        if index_type == 'hnsw':
            near_vector["ef"] = ef_search  # Set the ef parameter within near_vector

        # Perform the search using Weaviate
        search_query = client.query.get(collection_name, ["vector_id"]) \
            .with_near_vector(near_vector) \
            .with_limit(limit)

        response = search_query.do()
        if response is None or 'data' not in response or 'Get' not in response['data'] or collection_name not in response['data']['Get']:
            print(f"No results found for query: {query}")
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
            }
        ],
        "vectorizer": "none"  # Since we're providing precomputed vectors
    }

    client.schema.create_class(schema)
    vector_ids = list(range(len(base_vectors)))
    inserting_time = insert_vectors(base_vectors, vector_ids)

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
                    inserting_and_indexing_time = insert_vectors(base_vectors, vector_ids)

                    for limit in limit_values:
                        experiment_key = (experiment_type, m, ef_construct, ef_search, limit)
                        if experiment_key in completed_experiments:
                            pbar.update(1)
                            continue

                        result_ids, search_time = perform_searches(query_vectors, limit, ef_search, experiment_type)

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
