
import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    return the scalar dor product of the two vectors.
    # Hint: use `np.dot`.
    '''
    ### YOUR CODE HERE

    return np.dot(v1, v2)
    
def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    
    # Note: The cosine similarity is a commonly used similarity 
    metric between two vectors. It is the cosine of the angle between 
    two vectors, and always between -1 and 1.
    
    # The formula for cosine similarity is: 
    # (v1 dot v2) / (||v1|| * ||v2||)
    
    # ||v1|| is the 2-norm (Euclidean length) of the vector v1.
    
    # Hint: Use `dot_product` and `np.linalg.norm`.
    '''

    dot_prod = dot_product(v1, v2) 
    v1Norm =  np.linalg.norm(v1)
    v2Norm = np.linalg.norm(v2)
    normProd = v1Norm * v2Norm

    return dot_prod / normProd

    ### YOUR CODE HERE
    
def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    
    # Hint: You should use the cosine_similarity function that you already wrote.
    # Hint: For this lab, you can just use a for loop to iterate through vectors.
    '''
    ### YOUR CODE HERE

    closestSimilarity = -1
    index = 0

    for i, currVector in enumerate(vectors):
        similarity = cosine_similarity(target_vector, currVector)

        if similarity > closestSimilarity:
            closestSimilarity = similarity
            index = i

    return index