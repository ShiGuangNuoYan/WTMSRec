# ICBDSC 2025

## Dataset

We extracted the data set as [UniSRec](https://github.com/RUCAIBox/UniSRec) did。

## OverView

![](https://agent-demo-leo.oss-cn-chengdu.aliyuncs.com/%E5%B9%BB%E7%81%AF%E7%89%871.PNG)



Our Pipeline is shown above. The wavelet attention module is shown in the figure below:

![](https://agent-demo-leo.oss-cn-chengdu.aliyuncs.com/%E5%B9%BB%E7%81%AF%E7%89%872.PNG)

## Start 

If you want to use our model, follow the commands below:

```shell
pip install requirements.txt
cd WTMSRec
python Dis_Pretrain.py
```

The trained model is saved in the save folder.

```shell
python Dis_finetune.py
```

