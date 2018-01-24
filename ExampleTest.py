
from CheckTensors import *
from GeneralRelativity.Test_ComputeSpacetime import *

for dim in range(1, 4):
    print(printRank2TensorEqualitySymm("psi", compute_spacetime_metric(dim)))
