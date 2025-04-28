# Context Aware Recommendation Systems
Is time all that matters? Exploring Context-Aware-Recommendation-Systems to exploit novelty and diversity.

## Repository organization
```
Context-Aware-Recommendation-Systems/
├── Data/
│   ├── domain_transfer_capabilities/
│   ├── exploit_item_context/
│   ├── generalization_performance/
│   ├── increase_diversity_in_rec/
│   └── original_Data/
├── Experiments/
│   ├── domain_transfer_capabilities/
│   ├── exploit_item_context/
│   ├── generalization_performance/
│   └── increase_diversity_in_rec/
├── LICENSE
└── README.md
```

The data project contains the original data, which is the Amazon 2023 reviews dataset. It also has data files for the four experiments in this project. The data files have been separated for the experiments to avoid inadvertent overwriting of files by experiments. However, they all use the same original files which is close to 10GB and is read-only in our setup.     
     
The notebooks and code for the four experiments can be found in the `Experiments` folder and they are:     
#### 1. Domain Transfer Capabilities
The performance is evaluated under two settings to test SLi-Rec’s adaptability to new domains.     
       
#### 2. Exploit Item Context     
This experiment integrates a graph-based module using a Graph Convolutional Network (GCN) into the SLi-Rec architecture cite the DNN diagram and say where to enhance recommendation performance in scenarios where temporal signals alone may be insufficient.     
      
#### 3. Generalization Performance     
This experiment aims to measure the robustness of the model under different handicaps, while using the full validation and testing datasets.       
       
#### 4. Increase Diversity in Recommendation     
This experiment aims to exploit the negative sampling based on popularity to perform recommendations through a stochastic approach.     
      

## Preparing the environment, please follow these steps:

1. Install gcc if it is not installed already. 
1. Create and activate a new conda environment
```
    conda create -n ‹environment_name> python=3.9
    conda activate ‹environment_name>
```
1. Install the core recommenders package. It can run the CPU notebooks.
```
pip install recommenders
```
1. create a Jupyter kernel
```
python -m ipykernel install --user -name ‹environment_name> -display-name ‹kernel_name>
```
1. Install Tensorflow
```
pip install tensorflow==2.15.0
```
1. If using Mac with the Apple Silicon chip, you can optionally install Tensorflow Metal to optimize the use of Mac GPUs
```
pip install tensorflow-metal
```
1. Clone this repository within VSCode or using command line:
```
git clone https://github.com/recommenders-team/recommenders.git
```


## The initial data
To run the Python Notebooks, you need the 2023 Amazon reviews dataset at https://amazon-reviews-2023.github.io/ to be in the `Data > Original_Data` folder. However, you don't have to download it yourself as the code will download it if it does not find it in the folder. The size if this dataset is close to 10GB so please give it some time to download if it's your first run.       
       


## Running the Experiments in this project
Within VSCode:        
a. Open a main Notebook in one of the experiments as listed above          
b. Select Jupyter kernel «kernel_name>       
C. Run the notebook.       
      
