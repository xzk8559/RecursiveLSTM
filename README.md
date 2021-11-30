# RecursiveLSTM
**Recursive long short-term memory network for predicting nonlinear structural seismic response**

*Zekun Xu, Jun Chen(CA), Jiaxu Shen, Mengjie Xiang*

Artificial neural networks have been used to predict nonlinear structural time histories under seismic excitation because they have a significantly lower computational cost than the traditional time-step integration method. However, most existing techniques require simplification procedures such as downsampling to maintain identical length and sampling rates, and they lack sufficient accuracy, generality, or interpretability. In this paper, a recursive long short-term memory (LSTM) network was proposed for predicting nonlinear structural seismic responses for arbitrary lengths and sampling rates. Referring to the traditional integral solution method, the proposed LSTM model uses the recursive prediction principle and is therefore applicable to structures and earthquakes with different spectral characteristics and amplitudes. The measured ground motions and multilayer frame structures were used for model training and validation. The rules of hyperparameter selection for practical applications are herein discussed. The results showed that the proposed recursive LSTM model can adequately reproduce the global and local characteristics of the time history responses on four different structural response datasets, exhibiting good accuracy and generalization capability.

## Dependencies
- Python 3
- Tensorflow 2.0+

## Use
- If you want to experiment with datasets in paper：
    1. run `train.py` to train LSTM models.
    2. run `test.py` to evaluate on test sets.
- If you want to use your own datasets, refer to `train_MRF.py`, modify the data paths and reading process.

## Citation
If you use this code for your research, please cite our paper:

```
Zekun Xu, Jun Chen, Jiaxu Shen, Mengjie Xiang,
Recursive long short-term memory network for predicting nonlinear structural seismic response,
Engineering Structures,
Volume 250,
2022,
113406,
ISSN 0141-0296,
https://doi.org/10.1016/j.engstruct.2021.113406.
(https://www.sciencedirect.com/science/article/pii/S0141029621015133)
```
