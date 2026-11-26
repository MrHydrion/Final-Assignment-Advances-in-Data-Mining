import argparse
import sys
import random
import numpy as np
from scipy import sparse
from collections import defaultdict
from itertools import combinations
import psutil
import time
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs="?", default=3, help="Random seed (default: 3)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppres additional comments")
    parser.add_argument("-o", "--output", type=str, default="result.txt", help="Location output file (default: result.txt)")
    return parser.parse_args()

def loading_data(data_file):
    """
    it has 3 columns
    1) user_id
    2) movie_id
    3) rating
    
    """
    data = np.load(data_file).astype(np.int32, copy=False)
    print(f" maximum movie ID {max(data[: , 1])}")
    return data

def convert_to_sparse(data):
    """
    For our purpose, the ratings are not really necessary
    this function convert the np to a sparse matrix


    Rows: Movies
    Columns: User
    Value is 1 if movie was revied by user
    Coords: (Movie, Column),  Values: 1.0 or 0.0
    
    """
    # remove the last column
    data_usefull = np.delete(data, 2, 1)
    # -1 so we have 0 -> id 0
    row = (data_usefull[:, 1] - 1).astype(np.int32, copy=False)
    column = (data_usefull[:, 0] - 1).astype(np.int32, copy=False)

    # data needs to be fully 1
    value = np.ones(len(row), dtype=np.float32)

    csr = sparse.csr_matrix((value, (row, column)), dtype=np.float32)
    # print(type(csr))
 
    # print(csr[22,:].indices)
    # print(csr[22,:])

    # print(csr[0 , :]) 
    # the form is the ( user_id , movie_id ) and can something from the a user can be accesed with  print(csr[1, :]) # to acces the movies rater by the first person
    # we can also try out other sparse functions

    return csr


def minhash(csr, n, seed):
    """
    n = the number of rows and signatures
    
    return it in matrix form
    a column = 1 user_id 
    a row = 1 iteration of minihash (represented by n, where n should be bigger than 0)
    """

    rng = np.random.default_rng(seed)

    minhash_matrix =  np.full((n, csr.shape[1]), -1, dtype=np.int32) # -1 isnt an index

    for i in range(n):
        remaining = minhash_matrix.shape[1]
        current_iter = 0
        permutation = rng.permutation(csr.shape[0]).astype(np.int32, copy=False) # get a random permutation of size: amount of movies
        while remaining > 0 and current_iter < len(permutation):
            current_idx = permutation[current_iter] # get the idx of the permutation
            csr_idxes = csr.indices[csr.indptr[current_idx] : csr.indptr[current_idx + 1]]
            if csr_idxes.size > 0:
                unassigned = csr_idxes[minhash_matrix[i, csr_idxes] == -1]
                if unassigned.size > 0:
                    minhash_matrix[i, unassigned] = current_idx
                    remaining -= unassigned.size
            current_iter += 1

    return minhash_matrix


def LSH(signatures, b, r):
    """
    b = bands
    r = rows

    return pair candidates as a set
    """

    candidates = set()
    num_users = signatures.shape[1]

    for band in range(b):
        # extract the band from the signatures
        start = band * r
        end = start + r
        band_slice = signatures[start:end, :]

        buckets = defaultdict(list)
        for user_id in range(num_users):
            # hash the band into a bucket
            bucket_key = band_slice[:, user_id].tobytes()
            buckets[bucket_key].append(user_id)

        for users in buckets.values():
            # skip large buckets (>100) to avoid bottlenecks (hint 4)
            if 1 < len(users) <= 100:
                # take all combinations of users from this band and add to the candidates
                for u1, u2 in combinations(users, 2):
                    candidates.add((u1, u2) if u1 < u2 else (u2, u1))

    return candidates


def verify_and_save(sparse_csr, candidates, output_file="result.txt"):
    """
    writes the candidates that have a Jaccard Similarity higher than 0.5 to the output file
    """
    threshold = 0.5
    # convert to CSC
    csc = sparse_csr.tocsc()

    # clear output file
    open(output_file, "w").close()

    found_count = 0
    for u1, u2 in candidates:
        movies_1 = csc.indices[csc.indptr[u1]: csc.indptr[u1 + 1]]
        movies_2 = csc.indices[csc.indptr[u2]: csc.indptr[u2 + 1]]

        # find the ratio of the intersection (shared movies) and union (combined movies) (Jaccard Similarity)
        intersection = len(np.intersect1d(movies_1, movies_2, assume_unique=True))
        union = len(movies_1) + len(movies_2) - intersection

        # save if similarity is over 0.5
        if union > 0 and (intersection / union) > threshold:
            found_count += 1
            # open and close file for every pair (hint 6)
            with open(output_file, "a") as f:
                f.write(f"{u1 + 1},{u2 + 1}\n")

    return found_count


def main():
    total_start = time.perf_counter()
    peak_ram_mb = 0

    args = parse_args()
    seed = args.seed
    comments = not args.quiet
    t0 = time.perf_counter()
    data = loading_data("Data/user_movie_rating.npy")
    elapsed_time = time.perf_counter() - t0
    current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) # RAM in MB
    peak_ram_mb = max(peak_ram_mb, current_ram)
    if comments:
        print(f"[Load data] Time: {elapsed_time:.2f}s | Current RAM: {current_ram:.1f} MB | Peak RAM: {peak_ram_mb:.1f} MB")

    t0 = time.perf_counter()
    sparse_data = convert_to_sparse(data)
    elapsed_time = time.perf_counter() - t0
    current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) # RAM in MB
    peak_ram_mb = max(peak_ram_mb, current_ram)
    if comments:
        print(f"[Convert to sparse] Time: {elapsed_time:.2f}s | Current RAM: {current_ram:.1f} MB | Peak RAM: {peak_ram_mb:.1f} MB")
    del data # free ram

    t0 = time.perf_counter()
    n = 150 # 125
    minhash_matrix = minhash(sparse_data, n, seed)
    elapsed_time = time.perf_counter() - t0
    current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) # RAM in MB
    peak_ram_mb = max(peak_ram_mb, current_ram)
    if comments:
        print(f"[Minhash] Time: {elapsed_time:.2f}s | Current RAM: {current_ram:.1f} MB | Peak RAM: {peak_ram_mb:.1f} MB")


    t0 = time.perf_counter()
    b = 30 # 25
    r = 5 # 5
    # 5 * 25 = 125 (n)
    candidates = LSH(minhash_matrix, b, r)
    if comments:
        print(f"Candidates: {len(candidates)}")
    elapsed_time = time.perf_counter() - t0
    current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) # RAM in MB
    peak_ram_mb = max(peak_ram_mb, current_ram)
    if comments:
        print(f"[LSH] Time: {elapsed_time:.2f}s | Current RAM: {current_ram:.1f} MB | Peak RAM: {peak_ram_mb:.1f} MB")


    t0 = time.perf_counter()
    found = verify_and_save(sparse_data, candidates, output_file=args.output)
    if comments:
        print(f"Verified pairs: {found}")
    elapsed_time = time.perf_counter() - t0
    current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) # RAM in MB
    peak_ram_mb = max(peak_ram_mb, current_ram)
    if comments:
        print(f"[Verify & save] Time: {elapsed_time:.2f}s | Current RAM: {current_ram:.1f} MB | Peak RAM: {peak_ram_mb:.1f} MB")


    total_time = time.perf_counter() - total_start
    print(f"\nTotal found valid pairs: {found} ")
    print(f"Total runtime: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Observed peak RAM: {peak_ram_mb:.1f} MB ({peak_ram_mb / 1024:.2f} GB)")


if __name__ == "__main__":
    main()