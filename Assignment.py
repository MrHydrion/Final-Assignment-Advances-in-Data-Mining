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
    return data

def convert_to_sparse(data):
    """
    For our purpose, the ratings are not really necessary
    this function convert the np to a sparse matrix
    
    """
    # remove the last column
    data_usefull = np.delete(data, 2, 1)
    row = data_usefull[:, 1]
    column = data_usefull[:, 0]
    # data needs to be fully 1
    data = np.ones(len(row))

    csr = sparse.csr_matrix((data, (row, column)))
    # the form is the ( user_id , movie_id ) and can something from the a user can be accesed with csr [userid + 1 ,:]
    print(csr[1, 2]) 


def main():
    seed = 3
    data = loading_data("Data/user_movie_rating.npy")
    convert_to_sparse(data)
    pass


if __name__ == "__main__":
    main()