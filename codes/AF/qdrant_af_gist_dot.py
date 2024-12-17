from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np
import pandas as pd
import pickle
import csv
import time
from tqdm import tqdm
import os

# Load vectors from pickle file
def load_vectors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors


# Load the base and query vectors from pickle files
base_vectors_path = '../../data/gist_af_base_vector.pkl'
query_vectors_path = '../../data/gist_af_query_vector.pkl'
ground_truth_path = '../../data/gist_af_gt_dot.pkl'

base_vectors_with_attributes = load_vectors_from_pickle(base_vectors_path)

#print(base_vectors_with_attributes)

query_vectors_with_attributes = load_vectors_from_pickle(query_vectors_path)

base_vectors = list(base_vectors_with_attributes['vector'])
query_vectors = list(query_vectors_with_attributes['vector'])

print(f"Total vectors in dataset: {len(base_vectors_with_attributes)}")

print('Data loaded.')

# Qdrant setup
client = QdrantClient(url="http://localhost:6333", timeout=10000000)
collection_name = "trial_dot"
vector_size = 960
batch_size = 1000

# Experiment configuration
m_values = [8, 16, 32, 64]
#m_values = [64]
ef_construction_values = [64, 128, 256, 512]
#ef_construction_values = [512]
limit_values = [1, 10, 100]
ef_search_values = [128, 256, 512]
#ef_search_values = [512]
distance_metric = Distance.DOT

results_file = 'results/qdrant_attribute_filtering_GIST_dot.csv'


# Track completed experiments
completed_experiments = set()
if os.path.exists(results_file):
    with open(results_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            completed_experiments.add(tuple(row[:4]))

# Insert vectors once for all experiments

def insert_vectors(client, collection_name, base_vectors_with_attributes, batch_size):
    # Batch points listesi oluşturma
    batch_points = [
        PointStruct(
            id=i,
            vector=elem["vector"],
            payload={"attr1": elem["attr1"], "attr2": elem["attr2"], "attr3": elem["attr3"]}
        )
        for i, elem in base_vectors_with_attributes.iterrows()
    ]
    
    # Zaman ölçümünü başlat
    start_time_insert = time.time()
    
    # Batch halinde verileri ekleme
    num_batches = len(base_vectors) // batch_size + int(len(base_vectors) % batch_size > 0)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(base_vectors))
        batch_points_batch = batch_points[start_idx:end_idx]

        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=batch_points_batch
        )
    
    # Zaman ölçümünü bitir
    end_time_insert = time.time()
    print(f"Vectors inserted in {end_time_insert - start_time_insert:.2f} seconds.")
    return end_time_insert - start_time_insert



# Perform searches
def perform_searches(query_vectors_with_attributes, client, collection_name, ef, limit):
    result_ids = []
    start_time_search = time.time()
    for _, elem in query_vectors_with_attributes.iterrows():
        vec = elem["vector"]
        attr1, attr2, attr3 = elem["attr1"], elem["attr2"], elem["attr3"]
        #print(f"Query attributes: attr1={attr1}, attr2={attr2}, attr3={attr3}")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=vec,
            search_params=models.SearchParams(hnsw_ef=ef, exact=False),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="attr1", match=models.MatchValue(value=attr1)),
                    models.FieldCondition(key="attr2", match=models.MatchValue(value=attr2)),
                    models.FieldCondition(key="attr3", match=models.MatchValue(value=attr3))
                ]
            ),
            limit=limit,
        )
        result_ids.append([res.id for res in search_result])
    end_time_search = time.time()
    return result_ids, end_time_search - start_time_search

# Main experiment loop
with open(ground_truth_path, 'rb') as f:
    ground_truth = pickle.load(f)

total_iterations = len(m_values) * len(ef_construction_values) * len(limit_values) * len(ef_search_values)

with tqdm(total=total_iterations, desc="Experiment Progress") as pbar:

    # Initial collection creation
    if collection_name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_metric)
    )

    # Insert vectors once
    inserting_time = insert_vectors(client, collection_name, base_vectors_with_attributes, batch_size)

    # Koleksiyona eklenen vektörleri kontrol edin
    print(f"Total vectors in collection: {client.count(collection_name).count}")

    # Ground truth ve koleksiyon verilerini kontrol edin
    #print(f"Ground truth IDs: {ground_truth[:10]}")
    #vectors_in_collection, _ = client.scroll(collection_name=collection_name, limit=10)
    #for v in vectors_in_collection:
    #    print(f"Vector ID: {v.id}, Payload: {v.payload}")
    

    for m in m_values:
        for ef_construct in ef_construction_values:
            # Recreate collection with new HNSW parameters
            start_time_index = time.time()
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT)
            )
            insert_vectors(client, collection_name, base_vectors_with_attributes, batch_size)
            end_time_index = time.time()
            indexing_inserting_time = end_time_index - start_time_index

            for limit in limit_values:
                for ef_search in ef_search_values:
                    experiment_key = (m, ef_construct, limit, ef_search)
                    if experiment_key in completed_experiments:
                        pbar.update(1)
                        continue
                    
                    # Perform searches
                
                    result_ids, search_time = perform_searches(query_vectors_with_attributes, client, collection_name, ef_search, limit)

                    # Calculate recall
                    true_positives = 0
                    n_classified = 0
                    for i, elem in enumerate(result_ids):
                        true_positives_iter = len(np.intersect1d(ground_truth[i][:limit], result_ids[i]))
                        if true_positives_iter == 0:
                            print(f"No true positives for query {i}: ground truth {ground_truth[i][:limit]}, result_ids {result_ids[i]}")
                        true_positives += true_positives_iter
                        n_classified += len(elem)

                    #print(f"Result IDs: {result_ids}")
                    #print(f"Number of classified items (n_classified): {n_classified}")

                    recall = true_positives / n_classified

                    # Log results
                    avg_latency = search_time / len(query_vectors)
                    qps = len(query_vectors) / search_time
                    with open(results_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        if file.tell() == 0:
                            writer.writerow(['M', 'Ef Construct', 'Limit', 'Ef Search',
                                              'Avg Latency', 'Throughput (QPS)', 'Recall', 'Indexing Time', 'Inserting Time', 'Search Time'])
                        writer.writerow([m, ef_construct, limit, ef_search,
                                         avg_latency, qps, recall, indexing_inserting_time, inserting_time, search_time])
                    pbar.update(1)

print(f"Experiment for {distance_metric} completed.")
