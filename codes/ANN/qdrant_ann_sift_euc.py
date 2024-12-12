import pickle
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
from tqdm import tqdm
import csv
import time
import os

def fvecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]  # Get the dimension
    return a.reshape(-1, d + 1)[:, 1:].view('float32')  # Reshape and convert to float32

# Read vectors from the uploaded files
base_vectors = fvecs_read('../../data/sift_base_1m.fvecs')
query_vectors = fvecs_read('../../data/sift_query_1m.fvecs')

# Function to load vectors from a pickle file
def load_vectors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors

print('Data loaded.')

# Qdrant setup
client = QdrantClient(url="http://localhost:6922", timeout=10000000)
collection_name = "trial_euclid"
vector_size = 128
batch_size = 1000

# Experiment configuration
distance_metric = Distance.EUCLID
m_values = [8, 16, 32, 64]
ef_construction_values = [64, 128, 256, 512]
limit_values = [1, 10, 100]
ef_search_values = [128, 256, 512]

ground_truth_path = '../../data/sift_1m_groundtruth_euc.pkl'

# Track completed experiments
completed_experiments = set()
if os.path.exists('results/qdrant_ann_sift_results_euclid.csv'):
    with open('results/qdrant_ann_sift_results_euclid.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            completed_experiments.add(tuple(row[:4]))

# Insert vectors into the collection
def insert_vectors(client, collection_name, base_vectors, batch_size):
    start_time_insert = time.time()
    for batch_idx in range(0, len(base_vectors), batch_size):
        batch_vectors = base_vectors[batch_idx:batch_idx + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=list(range(batch_idx, batch_idx + len(batch_vectors))),
                vectors=batch_vectors
            )
        )
    end_time_insert = time.time()
    inserting_time = end_time_insert - start_time_insert
    print(f"Vectors inserted into collection '{collection_name}' in {inserting_time:.2f} seconds.")
    return inserting_time

# Perform searches
def perform_searches(query_vectors, client, collection_name, ef, limit):
    start_time_search = time.time()
    result_ids = []
    for query in query_vectors:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query,
            limit=limit,
            search_params=models.SearchParams(hnsw_ef=ef, exact=False)
        )
        result_ids.append([res.id for res in search_result])
    end_time_search = time.time()
    search_time = end_time_search - start_time_search
    return result_ids, search_time

# Main experiment loop
total_iterations = len(m_values) * len(ef_construction_values) * len(limit_values) * len(ef_search_values)

# Load ground truth for the current distance metric
with open(ground_truth_path, 'rb') as f:
    ground_truth = pickle.load(f)

with tqdm(total=total_iterations, desc="Experiment Progress") as pbar:
    # Create collection for the current distance metric
    if collection_name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_metric)
    )

    # Insert vectors once for this distance metric
    inserting_time = insert_vectors(client, collection_name, base_vectors, batch_size)

    for m in m_values:
        for ef_construct in ef_construction_values:
            # Recreate the collection with new parameters
            start_time_index = time.time()
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
                vectors_config=VectorParams(size=vector_size, distance=distance_metric)
            )
            insert_vectors(client, collection_name, base_vectors, batch_size)
            end_time_index = time.time()
            indexing_and_insert_time = end_time_index - start_time_index

            for limit in limit_values:
                for ef_search in ef_search_values:
                    experiment_key = (m, ef_construct, limit, ef_search)
                    if experiment_key in completed_experiments:
                        pbar.update(1)
                        continue

                    # Perform searches
                    result_ids, search_time = perform_searches(query_vectors, client, collection_name, ef_search, limit)

                    # Calculate recall
                    true_positives = 0
                    n_classified = 0
                    for i, elem in enumerate(result_ids):
                        true_positives_iter = len(np.intersect1d(ground_truth[i][:limit], result_ids[i]))
                        true_positives += true_positives_iter
                        n_classified += len(elem)
                    recall = true_positives / n_classified

                    # Print results for this configuration
                    avg_latency = search_time / len(query_vectors)
                    qps = len(query_vectors) / search_time

                    # Log results
                    with open('results/qdrant_ann_sift_results_euclid.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        if file.tell() == 0:
                            writer.writerow(['M', 'Ef Construct', 'Limit', 'Ef Search',
                                              'Avg Latency', 'Throughput (QPS)', 'Recall', 'Indexing Time', 'Inserting Time', 'Search Time'])
                        writer.writerow([m, ef_construct, limit, ef_search,
                                         avg_latency, qps, recall, indexing_and_insert_time, inserting_time, search_time])
                    pbar.update(1)

print("Experiment for EUCLID completed.")
