import sys
import random
import numpy as np
from scipy import sparse

def loading_data(data_file):
    """
    it has 3 columns
    1) user_id
    2) movie_id
    3) rating
    
    """
    data = np.load(data_file)
    print(f" maximum movie ID {max(data[: , 1])}")
    # print(data)
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
    row = data_usefull[:, 1] - 1 
    column = data_usefull[:, 0] - 1 

    # data needs to be fully 1
    value = np.ones(len(row))

    csr = sparse.csr_matrix((value, (row, column)))
    print(type(csr))
 
    print(csr[22,:].indices)
    print(csr[22,:])

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

    minhash_matrix =  np.full((n, csr.shape[1]), -1) # -1 isnt an index

    for i in range(n):
        remaining = minhash_matrix.shape[1]
        current_iter = 0
        permutation = rng.permutation(csr.shape[0]) # get a random permutation of size: amount of movies
        while remaining > 0 and current_iter < len(permutation):
            current_idx = permutation[current_iter] # get the idx of the permutation
            csr_idxes = csr[current_idx , :].indices
            unassigned = csr_idxes[minhash_matrix[i, csr_idxes] == -1]
            minhash_matrix[i, unassigned] = current_idx
            remaining -= len(unassigned)
                    
            current_iter += 1

    return minhash_matrix


def LSC(signatures, b, r):
    """
    b = bands
    r = rows
    
    """
    






    





def main():
    seed = 3
    data = loading_data("Data/user_movie_rating.npy")
    sparse_data = convert_to_sparse(data)
    minhash_matrix = minhash(sparse_data, 125, seed)
    pass


if __name__ == "__main__":
    main()