""" Collection of functions which print multidimensional numpy arrays as SpECTRE
Tensors, wrapped inside of a CHECK() macro, defined by the catch testing library. """

import numpy as np

def printScalarEquality(name: str,checkTensor: np.ndarray) -> str:
    returnString=""
    returnString+="  CHECK("+name+'.get() == approx('+str(checkTensor) + '));\n'
    return returnString+""


def printRank1TensorEquality(name: str,checkTensor: np.ndarray) -> str:
    dim=len(checkTensor)
    assert checkTensor.ndim==1
    returnString=""
    for i in range(0,dim):
        returnString+="  CHECK("+name+".get("+str(i)+') == approx('+str(checkTensor[i]) + '));\n'
    return returnString+""

def printRank2TensorEquality(name: str,checkTensor: np.ndarray) -> str:
    dim=len(checkTensor)
    assert checkTensor.ndim==2
    returnString=""
    for i in range(0,dim):
        for j in range(0,dim):
            returnString+="  CHECK("+name+".get("+str(i)+','+str(j)+') == approx('+str(checkTensor[i,j]) + '));'
    return returnString+""

def printRank2TensorEqualitySymm(name: str,checkTensor: np.ndarray) -> str:
    dim=len(checkTensor)
    assert checkTensor.ndim==2
    returnString=""
    for i in range(0,dim):
        for j in range(i,dim):
            returnString+="  CHECK("+name+".get("+str(i)+','+str(j)+') == approx('+str(checkTensor[i,j]) + '));\n'
    return returnString+""

def printRank3TensorEquality(name:str,checkTensor: np.ndarray) -> str:
    dim=len(checkTensor)
    assert checkTensor.ndim==3
    returnString=""
    for i in range(0,dim):
        for j in range(0,dim):
            for k in range(0,dim):
                returnString+="  CHECK("+name+".get("+str(i)+','+str(j)+','+str(k)+') == approx('+str(checkTensor[i,j,k]) + '));\n'
    return returnString+""

def printRank3TensorEqualitySymmIn2and3(name: str,checkTensor: np.ndarray) -> str:
    dim=len(checkTensor)
    assert checkTensor.ndim==3
    returnString=""
    for i in range(0,dim):
        for j in range(0,dim):
            for k in range(j,dim):
                returnString+="  CHECK("+name+".get("+str(i)+','+str(j)+','+str(k)+') == approx('+str(checkTensor[i,j,k]) + '));\n'
    return returnString+""
