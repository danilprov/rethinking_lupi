# Rethinking knowledge transfer in Learning Using Privileged Information (LUPI)

This code reproduces some of the experiments from 
[Lopez et al. 2016](https://github.com/lopezpaz/distillation_privileged_information) and [Collier et al. 2022](https://arxiv.org/abs/2202.09244), 
two papers that propose various knowledge transfer techniques for the LUPI paradygm.
Additionally, we conduct a real-life experiment using an open-source dataset from [IJCAI15 competition](https://ijcai-15.org/repeat-buyers-prediction-competition/).

## Synthetic experiments `synthetic/`
The notebooks `gen_dist_exp.ipynb`  and `tram_exp.ipynb` contain experiments with synthetic data for 
Generalized distillation [Lopez et al. 2016](https://github.com/lopezpaz/distillation_privileged_information) and TRAM [Collier et al. 2022](https://arxiv.org/abs/2202.09244).

## Experiments with stadardized datasets
### MNIST: scaling beyond `/mnist`
This code reproduces the MNIST experiment from [Lopez et al. 2016](https://github.com/lopezpaz/distillation_privileged_information) 
and extends its training epochs to beyond the original 50 epochs.
The code has been ported to work with Python 3.9, which required changing some of the requirements. 
For training, download the [MNIST dataset](https://yann.lecun.com/exdb/mnist/) and put it to `mnist/data`, `cd` to `mnist/` folder and run

```python mnist_varying_size.py```

### Sarcos: generalized distillation vs flat zero predictions `/sarcos`
This code reproduces the experiment from [Lopez et al. 2016](https://github.com/lopezpaz/distillation_privileged_information) 
and extends it with a naive baseline of predicting flat zeros. The code is from 2016 and requires Python 2.7.18 to run. 
Because the original code did not have requirements specified we have added the highest working requirements we could find. 
`sarcos/requirements.txt` contains the necessary dependencies.

For training, download the [Sarcos dataset](https://gaussianprocess.org/gpml/data/) and put it to `sarcos/data`, `cd` to `sarcos/` folder and run

```python sarcos.py```


## Real-world experiment with [IJCAI15 competition](https://ijcai-15.org/repeat-buyers-prediction-competition/) data `bandit_data/`
We compare Generalized distillation and TRAM on the Repeat Buyers Prediction dataset, a large-scale public dataset from the IJCAI-15 competition.
The data provides users’ activity logs of an online retail platform, including user-related features,
information about items at sale, and implicit multi-behavioral feedback such as click, add to cart, and purchase.

For training: 
1. download data from https://tianchi.aliyun.com/dataset/42
2. copy `user_info_format1.csv` and `user_log_format1.csv` to `bandit_data/data/IJCAI15/`
3. `cd bandit_data/`
4. run `python train.py`
