This repository holds the supplementary data files and source code for the following masterthesis:

# Title: Improving Semi-Structured Regression Models with a Novel Implementation in Torch

By combining additive models and neural networks, it becomes possible to expand the scope of statistical regression and extend deep learning-based approaches using interpretable structured additive predictors. The content of this thesis is the implementation of semi-structured distributional regression, a general framework that allows the combination of many different structured additive models and deep neural networks, using the R-native deep learning language torch. Using existing functionalities in the TensorFlow-based software R package [deepregression](https://github.com/neural-structured-additive-learning/deepregression) as a blueprint, this thesis will implement torch-based alternatives to existing functions.

The implementation of Torch-based SSDR models happened mainly in the [deepregression-masterthesis](https://github.com/marquach/DeepRegression.git) repository.

## Structure:

- `results` contains the results of the thesis:
  - `benchmarks`:
  In this analysis, four different approaches will be evaluated. The first approach utilizes [deepregression](https://github.com/neural-structured-additive-learning/deepregression) with TensorFlow as the engine, while the second employs a [Torch](https://github.com/mlverse/torch.git) model using a low-level loop approach. The third approach involves [Torch](https://github.com/mlverse/torch.git) with a high-level method, known as [Luz](https://github.com/mlverse/luz), and the fourth leverages [deepregression](https://github.com/neural-structured-additive-learning/deepregression) with [Torch](https://github.com/mlverse/torch.git) as the engine.
  - `numerical-experiments`: The numerical experiments are designed to compare the proposed framework with [Torch](https://github.com/mlverse/torch.git) as engine with traditional statistical regression frameworks, such as GAMs and GAMLSS.
  - `use-case`: Demonstrates the versatility of the [deepregression](https://github.com/neural-structured-additive-learning/deepregression) package but also shed light on its applicability to real-world data, such as the Airbnb dataset, and its ability to handle complex statistical models and deep learning approaches.
- `showcases.R` contains the code used for `numerical-experiments` and `use-case`.
- `speed_comparison` contains the code used for `benchmarks`.


# Related literature

The following works are based on the ideas implemented in this package:

* [Original Semi-Structured Deep Distributional Regression Proposal](https://arxiv.org/abs/2002.05777)
* [Neural Mixture Distributional Regression](https://arxiv.org/abs/2010.06889)
* [Deep Conditional Transformation Models](https://arxiv.org/abs/2010.07860)
* [Semi-Structured Deep Piecewise Exponential Models](https://arxiv.org/abs/2011.05824)
* [Combining Graph Neural Networks and Spatio-temporal Disease Models to Predict COVID-19 Cases in Germany](https://arxiv.org/abs/2101.00661)

# People that contributed

Many thanks to following people for helpful comments, issues, suggestions for improvements and discussions: 

* Dr. David Rügamer
* Chris Kolb
