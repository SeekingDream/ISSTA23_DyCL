This repository includes the implementation of our ISSTA 2023 paper [DyCL](https://github.com/SeekingDream/ISSTA23_DyCL/blob/main/CameraReady_ISSTA23_DyCL%20(20).pdf)

# Getting Started

### Installation



1. Setup the Python Virtual Environment
   1. Download ``conda``. First, run ``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``
      It will download script for installing the latest version of conda.
   2. Install ``conda``. Then run ``bash Miniconda3-latest-Linux-x86_64.sh`` for installation.  
   3. Create a virtual environment with ``conda create -n DLCL python=3.9``.
   4. Activate the created virtual environment with ``conda activate DLCL``

2. Install TVM 
    There are two ways to install TVM: (a) built from source; (b) install the pre-built binary.
   * (a) built from source. Please follow the instructions on their [website](https://tvm.apache.org/docs/install/from_source.html#install-from-source) to install from source code.
   * (b) install the pre-built binary. run ``pip install apache-tvm``

4. Install the Library
   1. Running ``pip install -r requirements_tx2.txt`` to install the necessary library



### File Structure
  * **adNN** -The code repository that implements different dynamic neural networks used in our evaluation.
    * **./adNN/blockdrop** - 
    * **./adNN/ImgCaption**
    * **./adNN/RANet**
    * **./adNN/ShallowDeep**
    * **./adNN/skipnet**
  * **src**  the core implementation of DyCL.
  * **compile_onnx.py** the Python script to compile DyNN with ONNXRuntime
  * **compile_tvm.py**  the Python script to compile DyNN with TVM
  * **evaluate_onnx.py** the Python script evaluates the accuracy of compiling with OnnxRuntime.
  * **evaluate_tvm.py** the Python script evaluates the accuracy of compiling with TVM.
  * **run_compile.sh** the bash script to compile our experimental DyNNs.
  * **run_evaluate.sh** the bash script to evaluate the correctness of our experimental DyNNs.
### How to Run
    
First compilerse the dynamic neural network model 
``bash run_compile.sh``

After compilation, evaluating the compiled executable
``bash run_evaluate.sh``



# Reusability

DyCL offers a highly automated and user-friendly experience. The current implementation allows us to compile any DyNN using either OnnxRuntime or TVM.


1. To demonstrate this, 
we can refer to the *load_model* function in ``utils.py``. 
In order to perform the compilation, 
we need to prepare the following components: 
the DyNN model instance ('model'), 
the source code of the model instance ('src'), 
the entry function for compilation ('compile_func'), 
an input demonstration of the DyNN ('example_x'), 
and a dataset for validating correctness ('test_loader').

2. Once these components are ready, 
we can customize the parameters for the compiler using the 'adnn_id' and compile it using either 'python compile_tvm.py --eval_id=adnn_id' or 'python compile_onnx.py --eval_id=adnn_id'."

