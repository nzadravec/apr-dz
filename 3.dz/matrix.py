import sys
import re
import copy

class Matrix:

    """this class makes it easier to handle two-dimensional
    matrix objects

    E.g. (m is object of class Matrix,
    a is object of class float)
    m = m * 5
    m = m * m
    m -= 1/2 * m
    m[1, 2]
    a = m[1,2]
    m[1, 2] = 4
    """
    
    eps = 1e-6

    # constructor
    # rows - number of rows
    # cols - number of columns
    # data - list containing lists of floats (E.g. [[1, 2][3, 4]])
    def __init__(self, rows, cols, data):
        
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.data = data

    def copy(self):
        return Matrix(self.rows, self.cols, copy.deepcopy(self.data))

    # overrides operator '+', creates new object
    # other - matrix
    # return - sum of self and other
    def __add__(self, other):
        data = list()
        for r in range(self.rows):
            row = list()
            for c in range(self.cols):
                row.append(self.data[r][c] + other.data[r][c])
            data.append(row)
        return Matrix(self.rows, self.cols, data)

    # overrides operator '+', result saves in self object
    # other - matrix
    # return - sum of self and other
    def __iadd__(self, other):
        return self + other

    def __iadd2__(self, other):
        for r in range(self.rows):
            for c in range(self.cols):
                self.data[r][c] += other.data[r][c]
        return self

    # overrides operator '-', result saves in self object
    # other - matrix
    # return - difference of self and other
    def __isub__(self, other):
        return self - other

    def __isub2__(self, other):
        for r in range(self.rows):
            for c in range(self.cols):
                self.data[r][c] -= other.data[r][c]
        return self

    # overrides operator '-', creates new object
    # other - matrix
    # return - difference of self and other
    def __sub__(self, other):
        data = list()
        for r in range(self.rows):
            row = list()
            for c in range(self.cols):
                row.append(self.data[r][c] - other.data[r][c])
            data.append(row)
        return Matrix(self.rows, self.cols, data)

    # overrides operator '*', creates new object
    # other - constant (number)
    # return - product of self and other
    def __rmul__(self, other):
        if isinstance(other, int):
            return Matrix.__mul__(self, other)


    # overrides operator '*', creates new object
    # other - matrix or constant (number)
    # return - product of self and other
    def __mul__(self, other):
        data = list()
        if isinstance(other, Matrix):
            for r in range(self.rows):
                row = list()
                for c in range(other.cols):
                    sum = 0
                    for k in range(0, self.cols):
                        sum += self.data[r][k] * other.data[k][c]
                    row.append(sum)
                data.append(row)
            return Matrix(self.rows, other.cols, data)

        else:
            for r in range(self.rows):
                row = list()
                for c in range(self.cols):
                    row.append(self.data[r][c] * other)
                data.append(row)
            return Matrix(self.rows, self.cols, data)

    # overrides operator '*', result saves in self object
    # other - matrix or constant (number)
    # return - product of self and other
    def __imul__(self, other):
        return self * other

    def __imul2__(self, other):
        for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] *= other
        return self

    # matrix transposition (A[i][j] = A[j][i]),
    # result saves in self object
    # return - transposed self
    def transpose(self):
        data = list()
        for r in range(self.cols):
                data.append(list())
        for c in range(self.cols):
                for r in range(self.rows):
                    data[c].append(self.data[r][c])
        return Matrix(self.cols, self.rows, data)

    # TODO
    def reshape(self, rows, cols):
            if self.rows*self.cols != rows * cols:
                raise Exception("...")
            else:
                _

    # sets value in matrix at position index
    # index - tuple (int, int)
    # value - type of float
    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.data[index][0] = value
        else:
            self.data[index[0]][index[1]] = value

    # gets value in matrix at position index
    # index - tuple (int, int) or int (treats self as vector)
    # return - value at position index
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index][0]
        else:
            return self.data[index[0]][index[1]]

    def __str__(self):
        string = ""
        for r in range(self.rows):
            for c in range(self.cols):
                string += "%10.2g" % (self.data[r][c])
                #string += "{:10.2f} ".format(self.data[r][c])
            string += "\n"
        return string

    # compares two matrices by their content
    # return - True if self and other are the same else False
    def __eq__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            return False

        for r in range(self.rows):
            for c in range(self.cols):
                if self.data[r][c] != other.data[r][c]:
                    return False
                
        return True

    # reades from file and creates it
    # definition of file: every row in file represents row in matrix,
    # values in single row are seperated by space or tab
    # return - new matrix defined in given file
    @staticmethod
    def create(fileName):
        file = open("lib/"+fileName, "r")
        lines = file.readlines()
        file.close()
        
        data = list()
        rows = 0
        #cols = len(lines[0].strip().split(" "))
        cols = len(re.split('\s+', lines[0].strip()))
        
        for l in lines:
            if len(l) == 1: continue # only \n
            
            rows += 1
            l = l[:-1] # remove \n
            l = l.strip()
            row = list()
            #for f in l.split():
            for f in re.split('\s+', l):
                row.append(float(f))
            data.append(row)
        return Matrix(rows, cols, data)

    # writes matrix to file
    def save(self, fileName):
        string = ""
        for r in range(self.rows):
            for c in range(1, self.cols):
                string += str(self.data[r][c])+" "
            string += "\n"

        file = open("lib/"+fileName, "w")
        file.write(string)
        file.close()

    # creates identity matrix
    # dim - dimension of matrix
    # return - identity matrix of dimension dim
    def eye(dim):
        rows = cols = dim
        data = list()
        for i in range(dim):
            row = list()
            for j in range(dim):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            data.append(row)
        return Matrix(rows, cols, data)

    # swaps two rows in matrix self
    # i - first row
    # j - second row
    def swapRows(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def lowerM(self):
        string = ""
        for r in range(self.rows):
            for c in range(self.cols):
                if r > c:
                    value = self.data[r][c]
                elif r == c:
                    value = 1
                else:
                    value = 0
                string += "{:7.2f} ".format(value)
            string += "\n"
        return string

    def upperM(self):
        string = ""
        for r in range(self.rows):
            for c in range(self.cols):
                if r <= c:
                    value = self.data[r][c]
                else:
                    value = 0
                string += "{:7.2f} ".format(value)
            string += "\n"
        return string

    @staticmethod
    def backwardSubst(A, b):
        n = A.rows
        y = b.copy()
        
        for i in range(n-1, -1, -1):

            if abs(A[i,i]) < Matrix.eps:
                    print("index: {}".format(i))
                    raise("divisor is zero!")
            
            y[i] /= A[i,i]
            for j in range(0, i):
                y[j] -= A[j,i] * y[i]

        return y

    @staticmethod
    def forwardSubst(A, b):
        n = A.rows
        y = b.copy()
        
        for i in range(n-1):
            for j in range(i + 1, n):
                y[j] -= A[j,i]*y[i]

        return y

    def LUDecomp(self, trace=False):
        if trace: print("lu decomposition:")
        n = self.rows
        A = self
        for i in range(n-1):

            if abs(A[i,i]) < Matrix.eps:
                    raise('pivot is zero!')

            if trace:
                print(A)
                print()
            
            for j in range(i+1, n):
                
                A[j,i] /= A[i,i]
                for k in range(i+1, n):
                    A[j,k] -= A[j,i] * A[i,k]

        if trace:
            print(A)
            print()
        
    def LUPDecomp(self, trace=False):
        if trace: print("lup decomposition:")
        n = self.rows
        A = self
        P = Matrix.eye(n)
        
        for i in range(n-1):

            if trace:
                print(A)
                print()
            
            pivot = i
            for j in range(i+1, n):
                if abs(A[j,i]) > abs(A[pivot,i]):
                    pivot = j
                    
            if abs(A[pivot,pivot]) < Matrix.eps:
                    raise('pivot is zero!')
                
            P.swapRows(pivot, i)
            self.swapRows(pivot, i)
            for j in range(i+1, n):
                A[j,i] /= A[i,i]
                for k in range(i+1, n):
                    A[j,k] -= A[i,k] * A[j,i]

        if trace:
            print(A)
            print()

        return P

    def inv(self):

        n = self.rows
        A = self.copy()
        inv = self.copy()
        P = A.LUPDecomp()
        e = Matrix.zeros(n, 1)

        for i in range(n):
            e[i] = 1.0
            y = Matrix.forwardSubst(A, P*e)
            x = Matrix.backwardSubst(A, y)
            e[i] = 0.0

            for j in range(n):
                inv[j,i] = x[j]

        return inv

    def zeros(rows, cols):
        data = list()
        for i in range(rows):
            row = list()
            for j in range(cols):
                row.append(0.0)
            data.append(row)

        return Matrix(rows, cols, data)

    def solve(self, b, p=True, trace=False, fileName=None):

        if fileName != None:
            orig_stdout = sys.stdout
            f = open(fileName, 'w')
            sys.stdout = f
        
        if trace:
            print("\tA * y = b")
            print("A:")
            print(self)
            print("b:")
            print(b)
        
        if p:
            P = self.LUPDecomp(trace)
            b = P*b
        else:
            self.LUDecomp(trace)
            
        if trace:
            print("\tA = L * U")
            print("L:")
            print(self.lowerM())
            print("U:")
            print(self.upperM())
            print("forward substitution:")
            print("\tL * (U * y) = b")
            print("\tL * x = b")
            x = Matrix.forwardSubst(self, b)
            print("x:")
            print(x)
            print("backward substitution:")
            print("\tU * y = x")
            y = Matrix.backwardSubst(self, x)
            print("y:")
            print(y)

            if fileName != None:
                sys.stdout = orig_stdout
                f.close()

            return y
        else:
            return Matrix.backwardSubst(self, Matrix.forwardSubst(self, b))
            
