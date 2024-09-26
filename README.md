# LLM Basics

This repository contains the implementation of encoder and decoder modules of the Transformer architecture. The encoder with classifier is utilized for classification tasks, while the decoder is used for language modeling. This project is a part of the SP24 CSE 256 - Statistical NLP course taught at UCSD.

## Code Structure

- **task_report**: This folder contains the problem statement for the project and the final analysis report.
- **Experiments.ipynb**: This Jupyter notebook contains the results and analyses of all three parts of the project.
- **llm**: This directory contains all internal code for implementing the Transformer, including utilities, tokenizers, etc.
- **results**: Contains result files generated during experimentation.

## llm

This folder contain the code files for all the implemented transformer architectures including the normal implementation and the AliBi implementation for Part 3. the AliBi Implementation Can be found in the file - transformer_exploration.py.

- Minor changes were made to the utilities.py file to accomodate the output of the transformer models.
- Major Changes to made tot the main.py file to include all 3 parts of the project.
- Minor Changes were made to the tokenizer.py file.
- No Chnages were made to the dataset.py file.

## Running the Code

To run the code, navigate to the project directory and execute the following command:

```bash
cd LLM
python3 llm/main.py task_type
```

Replace `task_type` with one of the following:

- **part1**: Task 1 - Classifier training (Part A)
- **part2**: Task 2 - Language Model training
- **part3a**: Task 3 - Architecture Exploration (AliBi Implementation)
- **part3b**: Task 3 - Performance Improvement (Encoder + Classifier) Exploration
- **part3b_best**: Task 3 - Performance Improvement - Best Model (Encoder + Classifier)

If you do not wish to run the code to check all the parts you can checkout the Experiments.ipynb notebook which contains the results and visualization of all the results obtained in each part.

## References

1. Vaswani, A., et al. (2017). "Attention is All You Need." arXiv:1706.03762.
2. Beltagy, I., et al. (2020). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." arXiv:2009.11304.

## Additional Resources

For a detailed explanation of the Transformer architecture, refer to Andrej Karpathy's [YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5101s).

Stanford CS224N: NLP with Deep Learning. [YouTube video](https://www.youtube.com/watch?v=5vcj8kSwBCY).
