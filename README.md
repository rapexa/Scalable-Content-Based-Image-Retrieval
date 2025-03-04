# Content-Based Image Retrieval (CBIR) Framework

## Overview
This project presents a **Hybrid Deep Hashing and Metric Space Partitioning Framework** for scalable **Content-Based Image Retrieval (CBIR)**. It integrates **unsupervised representation learning** with **VP-Tree optimization** to enhance retrieval accuracy and efficiency.

## Features
- **Autoencoder-Based Feature Extraction**: Generates compact and semantically meaningful image representations.
- **Hybrid Search Mechanism**: Uses Euclidean-based nearest neighbor search (O(N log N)) for moderate-scale datasets and **VP-Tree-based indexing (O(log N))** for large-scale retrieval.
- **Deep Hashing Techniques**: Enhances retrieval precision while maintaining computational efficiency.
- **Scalability and Speed Optimization**: Reduces search complexity without compromising retrieval accuracy.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

### Setup
Clone the repository and install dependencies:
```sh
git clone https://github.com/rapexa/Scalable-Content-Based-Image-Retrieval.git
cd Scalable-Content-Based-Image-Retrieval
pip install -r requirements.txt
jupyter lab .
```

## Results
The proposed model achieves **higher mean average precision (mAP), reduced search latency, and improved storage efficiency** compared to traditional CBIR techniques. Benchmark results on CIFAR-10 and ImageNet datasets demonstrate its superior performance.The experimental results validate the superiority of the proposed method, outperforming traditional CBIR techniques in terms of retrieval speed, accuracy, and computational efficiency. The proposed model achieves:

- 96.1% mAP
- 94.8% F1-Score
- 40% reduction in retrieval time compared to traditional models.

## Conclusion
This research provides a novel approach to CBIR systems, combining deep feature learning, adaptive hashing, and efficient indexing structures to deliver scalable, real-time image retrieval solutions for large datasets.

## Paper & Documentation
For an in-depth explanation, refer to the research paper:
[Download Paper (PDF)](./docs.pdf)

## Citation
If you use this work, please cite:
```bibtex
@article{your_paper,
  title={A Hybrid Deep Hashing and Metric Space Partitioning Framework for Scalable Content-Based Image Retrieval},
  author={S. Mohamadzadeh, M. Gharehbagh},
  journal={Your Journal},
  year={2025}
}
```

## License
This project is licensed under the MIT License.