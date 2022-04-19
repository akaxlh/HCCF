# HCCF

Environment:
```
python=3.6
tensorflow=1.14
```
Please unzip the datasets first. For Yelp, MovieLens and Amazon data, run the following commands, respectively:
```
python .\labcode_efficient.py --data yelp --temp 1 --ssl_reg 1e-4
python .\labcode_efficient.py --data ml10m --temp 0.1 --ssl_reg 1e-6 --keepRate 1.0 --reg 1e-3
python .\labcode_efficient.py --data amazon --temp 0.1 --ssl_reg 1e-7 --reg 1e-2
```
