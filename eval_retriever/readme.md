# Evaluation of Table Retriever

1. datasets

   - SPIDER(dev)
   - BIRD(dev)

   数据合计 2311条，可通过入参 `num` 控制测试的样本集大小。

2. files

   ```bash
   # code_gen_eval/eval_retriever/test_data
   ├── querys.json # 问题集
   ├── tables.json # 表格信息
   ├── y_columns.json # 字段召回标签
   └── y_tables.json # 表格召回标签
   ```

3. eval

   - 提供 `模型url`

     ```bash
     # python
     python eval_llm.py \
         --gen_model_url "http://localhost:8081" \
         --extract_model_url "http://localhost:8082" \
         --num 100
     # or bash
     bash run_llm.sh
     ```

   - 提供 `预测结果`

     ```bash
     # python
     python eval.py \
     	--pred_tables eval_retriever/preds/pred_tables.json \
     	--label_tables eval_retriever/test_data/y_tables.json \
     	--pred_columns eval_retriever/preds/pred_columns.json \
     	--label_columns eval_retriever/test_data/y_columns.json
     # or bash
     bash run.sh
     ```

4. response example

   ```json
   {
       "table": {
           "micro-Averaged Precision": 0.22109067017082787,
           "micro-Averaged Recall": 0.6735051288466349,
           "micro-Averaged F1": 0.33290051320101405,
           "macro-Averaged Precision": 0.13976550733152518,
           "macro-Averaged Recall": 0.3881846449171073,
           "macro-Averaged F1": 0.1857803816650417,
           "samples-Averaged Precision": 0.2676472224827915,
           "samples-Averaged Recall": 0.6640806906925472,
           "samples-Averaged F1": 0.35717170717066893,
           "weighted-Averaged Precision": 0.3637433649379988,
           "weighted-Averaged Recall": 0.6735051288466349,
           "weighted-Averaged F1": 0.4459455511007109,
           "Jaccard Similarity": 0.25583090223397414,
           "Hamming Loss": 0.020386645226267385
       },
       "column": {
           "micro-Averaged Precision": 0.024689698436169735,
           "micro-Averaged Recall": 0.19107617616460526,
           "micro-Averaged F1": 0.043729001878257186,
           "macro-Averaged Precision": 0.014774926641471512,
           "macro-Averaged Recall": 0.07186279322877256,
           "macro-Averaged F1": 0.020838020850031357,
           "samples-Averaged Precision": 0.03255324831448256,
           "samples-Averaged Recall": 0.1643324198278763,
           "samples-Averaged F1": 0.04558131269968282,
           "weighted-Averaged Precision": 0.098891750157333,
           "weighted-Averaged Recall": 0.19107617616460526,
           "weighted-Averaged F1": 0.11313546677422914,
           "Jaccard Similarity": 0.02539466616130353,
           "Hamming Loss": 0.015312494903495233
       }
   }
   ```

5. metric 

   在多分类问题中，评估指标的加权求平均方法主要有Macro、Micro、Weighted和Sample (Samples) 几种方式，它们主要应用于Precision、Recall、F1 Score等性能度量上。下面分别介绍这些方法及其特点，同时简要提及 Jaccard 和 Hamming Loss。

   ##### 5.1. Macro (宏平均)

   **定义**: Macro平均是先计算每个类别的Precision、Recall或F1 Score，然后对这些值取平均。这意味着每个类别在计算中被赋予了相同的权重，不管该类别样本数量多少。

   **公式**:

   ![image-20240528103235655](https://raw.githubusercontent.com/ryan-gz/img_cache/main/uPic/image-20240528103235655.png)

   **特点**: 强调每个类别的平等性，适用于类别不平衡的数据集，因为不会受大类别样本数量影响。

   **适用场景**: 当每个类别的错误都同等重要的时候，例如在某些医疗诊断中，即使是罕见病的诊断错误也不应被忽视。

   

   ##### 5.2. Micro (微平均)

   **定义**: Micro平均是将所有类别的TP、FP、FN加总后再计算整体的Precision、Recall或F1 Score。因此，样本数量较多的类别对结果影响较大。

   **公式**:

   ![image-20240528103300377](https://raw.githubusercontent.com/ryan-gz/img_cache/main/uPic/image-20240528103300377.png)

   **特点**: 更关注总体性能，受大类别影响较大，适合评估整体分类系统的效能。

   **适用场景**: 当关注的是模型的整体性能，特别是当数据类别分布不均时，Micro平均可以提供一个更全局的视角。

   

   ##### 5.3. Weighted (加权平均)

   **定义**: 加权平均考虑了每个类别的样本数量，即根据每个类别样本的比例来加权计算Precision、Recall或F1 Score。

   **公式**:

   ![image-20240528103316890](https://raw.githubusercontent.com/ryan-gz/img_cache/main/uPic/image-20240528103316890.png)

   **特点**: 能够反映类别不平衡的影响，适合于类别分布不均匀且希望类别重要性与样本数量成比例的场景。

   **适用场景**: 当不同类别的错误代价不同，且与类别大小相关时，Weighted平均更为合适。

   

   ##### 5.4. Samples-Weighted

   **定义**: 在多标签或多分类问题中，每个样本可能有多个标签，Sample-Weighted通常指的是按样本加权，考虑每个样本的真实标签和预测标签的差异。

   **特点**: 侧重于每个样本的贡献，而非类别的聚合表现。

   **适用场景**: 特别适合于多标签分类问题，强调个体样本的全面性和精确性。

   

   ##### 5.5. Jaccard Index (Jaccard相似系数)

   **定义**: 用于衡量两个集合交集大小与并集大小的比例，范围从0到1，值越接近1表示两个集合的相似度越高。

   **公式**:

   ![image-20240528103525622](https://raw.githubusercontent.com/ryan-gz/img_cache/main/uPic/image-20240528103525622.png)

   **特点**: 直观易懂，适用于二分类和多分类场景，尤其是集合重叠度的衡量。

   **适用场景**: 语义分割、信息检索等领域，衡量预测与真实标签的重叠程度。

   

   ##### 5.6. Hamming Loss

   **定义**: 在多标签分类中，Hamming Loss用来衡量预测标签集合与真实标签集合之间的差异，具体为误分类标签数占总标签数的比例。

   **公式**:

   ![image-20240528103433030](https://raw.githubusercontent.com/ryan-gz/img_cache/main/uPic/image-20240528103433030.png)

   其中N*N*是样本数，L*L*是标签数，II是指示函数，如果条件为真则为1，否则为0。

   **特点**: 直接反映了错误标签的比例，适合于多标签分类任务。

   **适用场景**: 当关心每个样本上具体哪些标签被错误预测时，Hamming Loss是一个合适的性能度量。