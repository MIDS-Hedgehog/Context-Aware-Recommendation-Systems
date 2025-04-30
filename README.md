# Context Aware Recommendation Systems
Is time all that matters? Exploring Context-Aware-Recommendation-Systems to exploit novelty and diversity.

## Repository organization
```
Context-Aware-Recommendation-Systems/
├── Data/
│   ├── domain_transfer_capabilities/
│   ├── context_aware_graph/
│   ├── generalization_performance/
│   ├── increase_diversity_in_rec/
│   └── original_data/
├── Experiments/
│   ├── domain_transfer_capabilities/
│   ├── exploit_item_context/
│   ├── generalization_performance/
│   └── increase_diversity_in_rec/
├── LICENSE
└── README.md
```

The `data` folder contains the `original_data` folder for the Amazon 2023 reviews dataset, as well as the data files for the four experiments. The data files have been separated for the experiments to avoid inadvertent file overwrites by experiments. However, they all use the same original files which is close to 10GB.     
     
The notebooks and code for the four experiments can be found in the `Experiments` folder and they are:     
#### 1. Domain Transfer Capabilities
The performance of the model is evaluated under two settings to test SLi-Rec’s adaptability to new domains.     
       
#### 2. context_aware_graph     
This experiment integrates a graph-based module using a Graph Convolutional Network (GCN) into the SLi-Rec architecture cite the DNN diagram and say where to enhance recommendation performance in scenarios where temporal signals alone may be insufficient.     
      
#### 3. Generalization Performance     
This experiment aims to measure how well the model generalizes under different data conditions, and thus the robustness.       
       
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


## The original data
To run the Python Notebooks, you need the 2023 Amazon reviews dataset at https://amazon-reviews-2023.github.io/ to be in the `Data > Original_Data` folder. You don't have to download it yourself as the code will download it if it doesn't find it in the folder. If this doesn't happen for any reason, please download and drop them in the folder. The size of this dataset is close to 10GB.       
       


## Running the Experiments in this project
Within VSCode:        
a. Open a main Notebook in one of the experiments as listed above          
b. Select Jupyter kernel «kernel_name>       
C. Run a notebook in any of the experiments.       
