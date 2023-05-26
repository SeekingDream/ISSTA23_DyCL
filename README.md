

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
   1. Running ``pip install -r requirements.txt`` to install the necessary library
   2. 

5. Install XXX


### File Structure
  * **adNN** -The code repository that implements different dynamic neural networks used in our evaluation.
    * **./adNN/blockdrop** - 
    * **./adNN/ImgCaption**
    * **./adNN/NMT**
    * **./adNN/RANet**
    * **./adNN/ShallowDeep**
    * **./adNN/skipnet**
  * **src**  TODO
  * **compile_onnx.py**
  * **compile_tvm.py**

### How to Run
    
First pacompilerse the dynamic neural network model 
``bash run_compil.sh``
After compilation, evaluating the compiled executable
``bash run_evaluate.sh``





