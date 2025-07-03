# AI-Flow-Ruyi (如意大模型)

<p align="center">
    <img src="assets/AI-Flow-Ruyi-logo.png" width="500" />
</p>

<p align="center">
        <a href="README.md">中文</a> &nbsp | &nbsp <a href="README_en.md">English</a>
        <br>
        🤗 <a href="">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑&nbsp <a href="">Paper</a>
</p>

#### Long long ago...
> 龙宫里藏着一根神棒，它能大能小，变化无穷。一日，龙王闲来无事，对着这根棒子念叨：“你这么厉害，要是能帮咱龙族干点别的事就好了。”棒子突然说话了：“我有个主意，这本事要是能用到帮人解决问题上...” 说干就干，棒子摇身一变，变成了一个超级厉害的大模型，能根据问题的难度，自由伸缩自己的“能力”。龙王一看，这不就是个能帮大家解决各种难题的“如意”宝贝嘛！于是取名“如意”，让它去人间帮忙了。

## 新闻

* 🎉🎉[2025/7/4]：如意-7B预览版（AI-Flow-Ruyi-7B-Preview）发布

## 介绍

**如意大模型（AI-Flow-Ruyi）** 是中国电信集团CTO、首席科学家、中国电信人工智能研究院 (TeleAI) 院长李学龙教授带领智传网（AI Flow）团队研发，是面向下一代“端-边-云”一体化网络架构的**同源家族模型（Familial Model）** 。其核心在于共享同源特征的早退出机制：模型能根据问题复杂度，调用不同参数规模的分支模型进行响应。各分支既可独立运行，又能依托同源特性实现信息共享与无缝切换，结合端-边-云分布式部署，实现推理效率的指数级提升，最终达成 “**基于连接与交互的智能涌现**”。

![](assets/ai-flow.png)
![](assets/ruyi_model.png)

## 如意-7B预览版

如意-7B预览版（AI-Flow-Ruyi-7B-Preview）于7月4日发布。其最大参数量分支为7B，可分化出具有等效参数量为3B、4B、5B、6B的早退出分支。其中：
* 3B、4B分支聚焦简单对话场景，其优势在于响应速度快、资源需求低；
* 5B、6B分支则针对日常通用任务场景，在性能与响应速度之间寻求平衡；
* 7B分支主要用于应对复杂问题，在多种能力维度上展现出较为全面的特性，但相对而言响应速度稍缓、资源需求略高。

|位点序号|早退出位置|等效模型大小|对应分支代号|场景定位|
|:-:|:-:|:-:|:-:|:-:|
|1|11层|3B|AI-Flow-Ruyi-7B-E3B|简单对话|
|2|15层|4B|AI-Flow-Ruyi-7B-E4B|简单对话|
|3|19层|5B|AI-Flow-Ruyi-7B-E5B|日常任务|
|4|23层|6B|AI-Flow-Ruyi-7B-E6B|日常任务|
|5|27层|7B|AI-Flow-Ruyi-7B-E7B|复杂问题|

### 训练过程

在训练开始前，我们基于Qwen团队预训练的[Qwen2.5-7B](https://arxiv.org/abs/2412.15115)模型（其已在18万亿高质量token上完成预训练），对7B主分支进行了参数初始化；对于早退出分支，其解码器层采用早退出位置的下一层参数进行初始化。

完成初始化后，我们采用**多分支联合预训练**方法，在私有高质量数据集上进行了约4000亿token的继续预训练，构建出如意-7B基座（AI-Flow-Ruyi-7B-Base）。

随后，我们基于约120万条高质量指令数据，对各分支进行了**联合指令遵循微调**，得到如意-7B预览版。

### 性能评测

我们基于[OpenCompass](https://github.com/open-compass/opencompass)及其官方配置文件，以0-shot方式在多个数据集上进行评测。评测结果表明，7B主分支在通用任务性能上与Qwen2.5-7B-Instruct基本持平。

<details>
<summary>通用任务评测</summary>

|模型名称|MMLU|MMLU-Pro|CMMLU|ARC-c|BBH|均分|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Qwen3-8B(think)|74.78|66.02|76.33|63.39|60.68|68.24|
|Qwen2.5-7B-Instruct|70.88|56.33|75.71|86.44|51.51|68.17|
|Llama-3.1-8B-Instruct|53.16|45.36|51.65|83.73|72.47|61.27|
|AI-Flow-Ruyi-7B-E7B<b>(ours)</b>|87.19|59.78|48.14|69.83|74.47|67.88|

</details>

<details>
<summary>代码任务评测</summary>

|模型名称|MBPP|HumanEval|LiveCodeBench|均分|
|:-:|:-:|:-:|:-:|:-:|
|Qwen3-8B(think)|78.60|84.76|63.10|75.49|
|Qwen2.5-7B-Instruct|70.82|84.15|34.55|63.17|
|Llama3.1-8B-Instruct|68.48|63.41|8.15|46.68|
|AI-Flow-Ruyi-7B-E7B<b>(ours)</b>|66.93|64.63|30.01|53.86|

</details>

<details>
<summary>STEM任务评测</summary>

|模型名称|Math|GPQA|GSM-8K|均分|
|:-:|:-:|:-:|:-:|:-:|
|Qwen3-8B(think)|83.84|38.38|93.03|71.75|
|Qwen2.5-7B-Instruct|73.66|35.35|88.48|65.83|
|Llama3.1-8B-Instruct|49.22|25.25|85.82|53.43|
|AI-Flow-Ruyi-7B-E7B<b>(ours)</b>|44.94|24.75|81.65|50.45|

</details>


同时，各早退出分支性能呈现出随等效参数量单调递增的趋势。

|模型名称|MMLU|MMLU-Pro|CMMLU|ARC-c|BBH|均分|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|AI-Flow-Ruyi-7B-E3B<b>(ours)</b>|66.93|44.70|19.80|40.00|32.29|40.74|
|AI-Flow-Ruyi-7B-E4B<b>(ours)</b>|78.86|48.60|26.51|58.98|41.98|50.99|
|AI-Flow-Ruyi-7B-E5B<b>(ours)</b>|75.34|49.13|33.91|65.76|64.48|57.72|
|AI-Flow-Ruyi-7B-E6B<b>(ours)</b>|84.58|53.06|33.94|73.22|47.33|58.43|
|AI-Flow-Ruyi-7B-E7B<b>(ours)</b>|87.19|59.78|48.14|69.83|74.47|67.88|

## 使用

Step 1. 克隆本仓库至本地

```
git clone https://github.com/TeleAI-AI-Flow/AI-Flow-Ruyi.git
```
