import numpy as np
from numpy.core.defchararray import array

def cosine_distance(matrix1,matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance

def cal_cosine_sim(a, b):
    zero_array = np.array([0] * len(a))
    if (a == zero_array).all() or (b == zero_array).all():
        return float(1) if (b == a).all() else float(0)
    res = np.array([[a[i] * b[i], a[i] * b[i], a[i] * b[i]] for i in range(len(a))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 

def cosine_distance(a, b):
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

# matrix1=np.array([[0,0]])
# matrix2=np.array([[2,1]])
# cosine_dis=cosine_distance(matrix1,matrix2)
# print (cosine_dis)

a = cosine_distance(np.array([[0,0.0001,0.33],[0,0.1,0.9]]), np.array([[0.1,0.9,0.11],[0.3,0.4,0.4]]))
print(a)