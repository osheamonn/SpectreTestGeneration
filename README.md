# SpectreTestGeneration
Python code designed to aid test creation for functions involving tensors in SpECTRE.

Tensors are created with 'random' values within the CreateTensors files. These values should match up exactly with the created tensors within SpECTRE. 

The computations which are being tested are defined in the 'Test_*' files within each directory. Use of np.einsum is encouraged. 

Requires python3 due to use of type annotations.

