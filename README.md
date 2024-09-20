[TOC]

# ICBDSC 2025

## Dataset

We extracted the data set as [UniSRec](https://github.com/RUCAIBox/UniSRec) did.

Create a folder named dataset under the root directory with the file structure as follows:

```
├── dataset
│   | downstream 
│     ├── Arts_mm_full
│     ├── Arts_mm_subset 
│     ├── Instruments_mm_full
│     ├── Instruments_mm_subset
│     ├── Office_mm_full
│     ├── Office_mm_subset
│     ├── Pantry_mm_full
│     ├── Pantry_mm_subset
│     ├── Scientific_mm_full
│     ├── Scientific_mm_subset
│   | pretrain
│     ├── FHCKM_mm_full
```



## OverView

![](https://agent-demo-leo.oss-cn-chengdu.aliyuncs.com/%E5%B9%BB%E7%81%AF%E7%89%871.PNG)



Our Pipeline is shown above. The wavelet attention module is shown in the figure below:

![](https://agent-demo-leo.oss-cn-chengdu.aliyuncs.com/%E5%B9%BB%E7%81%AF%E7%89%872.PNG)

## Start 

If you want to use our model, follow the commands below:

```shell
pip install -r requirements.txt
cd WTMSRec
python Dis_Pretrain.py
```

The trained model is saved in the save folder.

```shell
python Dis_finetune.py
```

