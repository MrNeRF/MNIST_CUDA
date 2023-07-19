
# MNIST Classifier with CUDA and C++

This project is a MNIST classifier using CUDA and C++ to code an MLP from scratch. 
In its tests it uses the torch C++ API to assure correct implementation. It achieves ~97% on MNIST dataset.

Attention: This not yet in a clean version, but it is working. It is not optimized at all. 

## Dependencies

- CMake (version >= 3.22)
- CUDA Toolkit (version >= 12.0)
- PyTorch (libtorch)
- Google Test (release-1.10.0)

## Installation

### libtorch

First, download the libtorch library using the following command:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
```

This will download a zip file named `libtorch-shared-with-deps-latest.zip`. To extract this zip file, use the command:

```bash
unzip libtorch-shared-with-deps-latest.zip -d external/
```

This will create a folder named `libtorch` in the `external` directory of your project.

### MNIST Data

The MNIST data can be downloaded using the following command:

```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

These commands will download four gzip files. To extract these gzip files, use the command:

```bash
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

Then, move the extracted files into the `data` directory of your project.

## Building the Project

To build the project, follow these steps:

1. Open a terminal and navigate to the project's root directory.
2. Create a new directory named `build` and navigate into it:

    ```bash
    mkdir build && cd build
    ```

3. Run the CMake configuration:

    ```bash
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```

4. Finally, compile the project:

    ```bash
    make -j$(nproc)
    ```

This will create an executable named `mnist_cuda` in the `build` directory.
    ```bash
    ./build/mnist_cuda data
    ```

## Running the Tests

After building the project, you can run the tests with the following command:

    ```bash
    ./build/cuda_kernel_tests
    ```
