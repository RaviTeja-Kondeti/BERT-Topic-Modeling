# BERT Topic Modeling

Advanced topic modeling using BERT embeddings and BERTopic for unsupervised text analysis and semantic clustering.

## Overview

This project implements state-of-the-art topic modeling techniques using BERT (Bidirectional Encoder Representations from Transformers) and the BERTopic library. By leveraging pre-trained transformer models, this approach captures semantic similarities and discovers coherent topics in text data with superior performance compared to traditional methods like LDA.

## Features

- **BERT-based Embeddings**: Utilizes pre-trained BERT models for generating contextual embeddings
- **Advanced Clustering**: Implements HDBSCAN for density-based clustering
- **Topic Representation**: Uses c-TF-IDF for generating interpretable topic descriptions
- **Dimensionality Reduction**: Employs UMAP for efficient high-dimensional data visualization
- **Interactive Analysis**: Provides comprehensive visualizations and topic exploration

## Methodology

### 1. Text Embedding
Text documents are converted into dense vector representations using pre-trained BERT models, capturing semantic meaning and context.

### 2. Dimensionality Reduction
UMAP (Uniform Manifold Approximation and Projection) reduces the high-dimensional BERT embeddings while preserving the local and global structure of the data.

### 3. Clustering
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) identifies dense regions in the reduced space, forming coherent topic clusters.

### 4. Topic Representation
c-TF-IDF (class-based Term Frequency-Inverse Document Frequency) generates descriptive keywords for each discovered topic.

## Installation

```bash
pip install bertopic
pip install sentence-transformers
pip install umap-learn
pip install hdbscan
```

## Usage

```python
from bertopic import BERTopic

# Initialize BERTopic model
model = BERTopic()

# Fit the model on your documents
topics, probabilities = model.fit_transform(documents)

# Get topic information
topic_info = model.get_topic_info()

# Visualize topics
model.visualize_topics()
```

## Results

The model successfully identifies meaningful topics in the dataset with high coherence and interpretability. The BERT-based approach outperforms traditional topic modeling methods by:

- Capturing semantic relationships between words
- Handling polysemy and context-dependent meanings
- Generating more coherent and interpretable topics
- Better handling of short texts and sparse data

## Project Structure

```
BERT-Topic-Modeling/
├── notebooks/
│   └── topic_modeling_analysis.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

## Technologies Used

- **BERTopic**: Topic modeling library
- **Sentence Transformers**: Pre-trained BERT models
- **UMAP**: Dimensionality reduction
- **HDBSCAN**: Density-based clustering
- **Python**: Programming language
- **Jupyter Notebook**: Interactive development environment

## Applications

- Document clustering and organization
- Content recommendation systems
- Trend analysis in text data
- Customer feedback analysis
- Social media monitoring
- Academic paper categorization

## Future Enhancements

- Dynamic topic modeling for temporal analysis
- Multi-lingual topic modeling
- Integration with domain-specific BERT models
- Real-time topic detection
- Topic evolution tracking

## References

- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

## License

MIT License

## Author

Ravi Teja Kondeti

## Acknowledgments

Built using the BERTopic library by Maarten Grootendorst and powered by transformer models from Hugging Face.
