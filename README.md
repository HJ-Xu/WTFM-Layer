# WTFM-Layer

Code for "WTFM Layer: An Effective Map Extractor for Unsupervised Shape Correspondence", published at PG 2022. 

You can view the detailed content of the paper [here](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14656).

![texture_compare_yesno.pdf](https://github.com/HJ-Xu/WTFM-Layer/files/11107903/texture_compare_yesno.pdf)

# Installation
```
python  >= 3.7
pytorch >= 1.12.0
```

# Download data
Regarding the [remesh 5K dataset](), we used the dataset from Geofmnet, and for the [anisotropic dataset](), we used the dataset from DUO. Please put the downloaded data in the off format into the directory `data/datasetname/shapes`.

# Usage
To train WTFM model, use the training script:
```
> python train.py  
```
To evaluate a trained model, use:
```
> python train.py --evaluate
```

# Citation
```
@inproceedings{liu2022wtfm,
  title={WTFM Layer: An Effective Map Extractor for Unsupervised Shape Correspondence},
  author={Liu, Shengjun and Xu, Haojun and Yan, Dong-Ming and Hu, Ling and Liu, Xinru and Li, Qinsong},
  booktitle={Computer Graphics Forum},
  volume={41},
  number={7},
  pages={51--61},
  year={2022},
  organization={Wiley Online Library}
}
```
