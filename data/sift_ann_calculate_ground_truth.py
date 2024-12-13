import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def calculate_distance(vector1, vector2, similarity_function='euclidean'):
    if similarity_function == 'euclidean':
        return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=2)
    elif similarity_function == 'cosine':
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    elif similarity_function == 'dot':
        return np.dot(vector1, vector2)
    else:
        raise ValueError("Unsupported similarity function")


def select_k_closest_elements(df, reference_vector, k, similarity_function='euclidean'):
    df = df.copy()
    df['distance'] = df['vector'].apply(lambda v: calculate_distance(v, reference_vector, similarity_function))
    if similarity_function == 'euclidean':
        result_df = df.nsmallest(k, 'distance').drop(columns=['distance'])
    else:
        result_df = df.nlargest(k, 'distance').drop(columns=['distance'])

    return result_df.index.values


def filter_by_attributes(qvec, bvecs):
    columns_to_match = bvecs.columns[1:]

    mask = (bvecs[columns_to_match] == qvec[columns_to_match]).all(axis=1)

    filtered_df = bvecs[mask]

    return filtered_df


def top_k_neighbors(query_vectors, base_vectors, k=100, function='euclidean', filtering=True):
    """
    Calculates the top k neighbors (ground truth), if filtering = True, we filter all boolean attributes as well.
    """
    if function not in ['euclidean', 'cosine', 'dot']:
        raise NotImplementedError("Other distance functions are not yet implemented")

    top_k_indices = []

    for _, elem in tqdm(query_vectors.iterrows(), total=query_vectors.shape[0], desc="Processing queries"):
        if filtering:
            filtered_df = filter_by_attributes(elem, base_vectors)
        else:
            filtered_df = base_vectors

        result = select_k_closest_elements(filtered_df, elem["vector"], k, similarity_function=function)
        top_k_indices.append(result)

    return top_k_indices


base_vectors = list(fvecs_read('sift_base.fvecs')) # gist_base.fvecs
query_vectors = list(fvecs_read('sift_query.fvecs')) # gist_query.fvecs

query_vectors = pd.DataFrame({'vector': query_vectors})
base_vectors = pd.DataFrame({'vector': base_vectors})

truth = top_k_neighbors(query_vectors, base_vectors, function='euclidean', filtering=False)
truth2 = top_k_neighbors(query_vectors, base_vectors, function='cosine', filtering=False)
truth3 = top_k_neighbors(query_vectors, base_vectors, function='dot', filtering=False)

with open('sift_1m_groundtruth_euc.pkl', 'wb') as file:
    pickle.dump(truth, file)

with open('sift_1m_groundtruth_COSINE.pkl', 'wb') as file:
    pickle.dump(truth2, file)

with open('sift_1m_groundtruth_IP.pkl', 'wb') as file:
    pickle.dump(truth3, file)

#base_vectors.to_pickle('Downloads/buyuk_base_vectors_with_attributes50k.pkl') # gist
#query_vectors.to_pickle('Downloads/buyuk_query_vectors_with_attribute50ks.pkl') # gist