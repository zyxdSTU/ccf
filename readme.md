#### 互联网金融新实体发现

>最近参加了一个命名实体识别比赛，学到了很多调参技巧，比赛进了复赛就没有打了
>
>开源一下自己的代码，以后自己可以参考参考。

#### 方法

主要的方法是使用``bert+lstm``, ``bert+idcnn``,``bert+lstm+attention``.

其中效果最优的是``bert+lstm+attention``

trick的话，主要就是在计算损失的时候，增加'B', 'I', 'E'标签损失比重，减少’O'标签损失比重

#### 目录结构

- data目录

  包含原始的Test_Data.csv、Train_Data.csv

  - 5-fold目录

    包含5折交叉验证的5份语料，0.txt、1.txt、2.txt、3.txt、4.txt

  - bilstm、bilstm_attn、idcnn是各个模型的运行结果目录

    包含result.txt, 0.csv,1.csv,2.csv,3.csv,4.csv,5.csv各份数据对应的测试结果

- data_util.py是形成训练数据和测试数据的脚本

- bilstm.py、bilstm_attn.py、idcnn.py是相应的模型文件

- 配置文件 config.yml

#### 运行方式

```shell
#bilstm可改为其他模型 idcnn、bilstm_attn
python3 ensemble.py -m bilstm
#结果在/data/5-fold/bilstm/
```

