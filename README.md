# 机器学习大作业--DCRec：基于去偏对比学习的序列推荐模型


本仓库包含了**DCRec（基于去偏对比学习的序列推荐模型）**的实现，该模型旨在通过解决**流行度偏差**和**数据稀疏性**问题来提升推荐性能。

## 本仓库目录:
```bash
├─data//数据集
│  ├─Grocery_and_Gourmet_Food
│  └─MovieLens_1M
│      ├─ml-1m
│      ├─ML_1MCTR
│      └─ML_1MTOPK
├─docs
│  ├─demo_scripts_results
│  ├─tutorials
│  └─_static
├─log//存放训练日志
│  └─Dcrec
└─src
    ├─helpers
    ├─models//存放模型
    │  ├─context
    │  ├─context_seq
    │  ├─developing 
    │  ├─general  
    │  ├─reranker
    │  ├─sequential//Dcrec在该目录下
    └─utils
```    
## 基于Rechorus框架的实现

在实现**DCRec**模型时，我对原始代码库进行了以下几项修改：

### 新增文件
- **`DcrecRunner`和`DcrecReader`文件**：更好地处理数据加载和模型训练流程
- **`model/Dcrec.py`文件**：包含模型架构实现和**自身改进**(课程学习和lightGCN)
- **`utils/dcrec_util.py`文件**：优化代码组织结构和模块化程度


### 环境配置及运行

由于dgl库在windows系统上暂停更新,所以需采用较低版本的pytorch,老师/助教您的numpy等库可能也需要**连带降级**,请注意这一点!!

```bash
pip install -r requirements.txt
```
```bash
cd Rechorus-Dcrec
cd src
python main.py --model_name Dcrec --epoch 20  --path ..\data\MovieLens_1M\ --dataset ML_1MTOPK   --lr 0.0015 --dropout 0.2
python main.py --model_name Dcrec --epoch 20  --path ..\data\ \ --lr 0.001 --dropout 0.2
```
请保证**安装好环境**并**在src目录下**运行main.py文件!!
运行成功后您应该可以看见有关参数信息出来,如果老师/助教您运行不了，可以联系我。

## 运行结果

 **Dcrec** 在 **Grocery and Gourmet Food** 和 **MovieLens 1M** 数据集上的. 下面是实验结果(epoch=20),baseline老师/助教您可以在课程报告上查看。

### Grocery and Gourmet Food 数据集

| Metrics      | HR@5  | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|--------------|-------|-------|-------|-------|--------|---------|---------|---------|
|Dcrec         |	0.4022  |	0.4889	|0.5951	|**0.7960** |	0.3102	|0.3382|	0.3650|	**0.4046**|
|Dcrec(课程学习)|	**0.4028**|	0.4887|	0.5949|	0.7934     |	**0.3107**|	**0.3385**|	**0.3652**|	0.4043|
|Dcrec(lightGcN)	|0.4022	|**0.4896**	|**0.5958**	|0.7937	|0.3102|	0.3384	|0.3651	|0.4041|

### Movielen-1m数据集（topk）

| Metrics      | HR@5  | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|--------------|-------|-------|-------|-------|--------|---------|---------|---------|
|Dcrec|	0.5170|	0.6496|	0.7811|	0.9113|	0.3916|	0.4347|	0.4679|	0.4938|
|Dcrec(课程学习)|	0.5146|	0.6514|	0.7704|	0.9040|	0.3852|	0.4295|	0.4596|	0.4862|
|Dcrec(lightGCN)|	**0.5313**|	**0.6653**|**0.7990**|	**0.9168**|	**0.3962**|	**0.4394**|	**0.4683**|	**0.4955**|
