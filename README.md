# About The Project
----
[![Supported Python versions](https://shields.mitmproxy.org/pypi/pyversions/mitmproxy.svg)](https://pypi.python.org/pypi/mitmproxy)

``BS_Opt`` 实现了对期权价格中封装的状态价格密度的非参数估计。我们建议直接使用非参数混合物对状态价格密度进行建模,并使用最小二乘法对其进行估计。

实现逻辑：
1. 先生成一个合理的 sigma 范围
2. 对每个 sigma 
3. 先进行一次二次规划得到 weight
4. 再使用牛顿迭代法得到μ
## Install
1. Clone the repo

```
git clone https://github.com/Phoenizard/BS_Opt.git
```

## Usage

``notebooks`` 中的 ``Simulated2.ipynb`` 是利用BS公式生成的模拟数据，``main`` 中为采用了 AMD 公司的真实数据，可以直接运行。
```
├── config
│   └── config.py
├── data
│   ├── BS_Theoretical.py
│   ├── data_grip.py
│   └── data_process.ipynb
├── image
│   ├── MixBS_Test_AMD.png
│   └── MixBS_Train_AMD.png
├── modules
│   ├── __pycache__
│   ├── BS_MixG_model.py
│   └── BS_Theoretical_Model.py
├── notebooks
│   └── Simulated2.ipynb
├── .gitignore
├── AMD_analysis.ipynb
├── main.py
├── model.py
├── optimize.py
└── utils.py

```

## Contributing

PRs accepted.

## License

MIT © Richard McRichface
