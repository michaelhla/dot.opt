
from sys import argv
import numpy as np
import time


def multiply(A, B, C, N):
    C = [N][N]
    for i in range(N):
        for j in range(N):
            C[i][j] = 0
            for k in range(N):
                C[i][j] += A[i][k]*B[k][j]
    return C


def split(matrix):
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]


def strassen(x, y):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """

    osize = x.shape[0]

    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return x * y

    if x.shape[0] % 2 != 0 or y.shape[0] % 2 != 0:
        rowadd = [np.zeros(x.shape[0], dtype=int)]
        coladd = np.zeros(x.shape[0]+1, dtype=int)
        x = np.append(x, rowadd, axis=0)
        y = np.append(y, rowadd, axis=0)
        x = np.hstack((x, np.atleast_2d(coladd).T))
        y = np.hstack((y, np.atleast_2d(coladd).T))

    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)

    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11[:osize][:osize], c12[:osize][:osize])), np.hstack(
        (c21[:osize][:osize], c22[:osize][:osize]))))

    return c


def main():
    n = 200
    A = np.ones((n, n), dtype=int)
    B = np.ones((n, n), dtype=int)
    begin = time.time()
    res = strassen(A, B)
    end = time.time()
    print(end-begin)
    print(res)
    # for k in range(n):
    #     print(res[k][k])


if __name__ == "__main__":
    main()
