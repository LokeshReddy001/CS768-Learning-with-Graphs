# Datasets:
- Cora, Citeseer, Pubmed, A-computer, and A-photo
- two large OGB datasets, Arxiv and Products
- Carefully see what the feautures are in the datasets and inputs and outputs to the models


# Model Architecture:
- Teacher Model - GraphSAGE + GCN aggragator. GCN aggregation is simply sum(h_v/root(degree(v))) from neighbors of v  + {v}. Inputs are the node content features of the nodes in the graph.
- Student Model - MLP with inputs as node content features and node position features.
- Need to see how the node position features are generated. paper mentions they used deepwalk to generate the node position features.

# Loss Function:
- Supervised Training - Objective is to preidct the labes of nodes
- Knowledge Distillation - sum over various loss components:
    1. Cross entropy loss between student outputs and ground truth labels
    2. KL divergence loss between student outputs and teacher outputs
    3. Representational Similarity Distillation - frobenius norm between NXN similarity matrix of student and teacher outputs
    4. *Adversarial Feature Augmentation - same as 1 and 2 but with feature augmented with noise (not fully clear of methodology here)*

# Benchmarking:
- Need to compare in both transductive and inductive settings
- *Need to see how nodes are selected for training and testing in inductive setting*
- Need to plot inference times vs accuracy for different baselines(NOSMOG, GLNN, GNN)
- Ablation study - plug and play different components of the loss function and see how it affects the performance of the model