CTGM is a deep learning framework for predicting interactions between circRNAs and miRNAs. It integrates graph convolutional networks (GCNs), gated mechanisms, and Transformer encoders to effectively combine sequence embedding features and structural features, achieving high-accuracy interaction prediction.
Environmental Requirements
Python 3.8+
PyTorch 1.9+
NumPy, Pandas, scikit-learn, Matplotlib
The project uses the CMI-9589 dataset by default (which includes 9,589 known circRNA-miRNA interactions). The dataset directory structure is as follows
Dataset/
└── CMI-9589/
    ├── Positive_Sample_Train0.csv
    ├── Negative_Sample_Train0.csv
    ├── Positive_Sample_Test0.csv
    ├── Negative_Sample_Test0.csv
    ├── ...
