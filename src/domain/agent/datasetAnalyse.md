## 此文件为数据集分析智能体说明

### 🧾工作流程
1. 制作副本并递归解压所有压缩包
2. 分析文件树结构并保存
3. 分析元数据补充文件树
4. 输入到大模型中增强文件树
5. 保存结果

### 👌目前支持元数据分析的文件类型
- ✅ `.jpg`
- ✅ `.jpeg`
- ✅ `.png`
- ✅ `.bmp`
- ✅ `.gif`
- ✅ `.csv`
- ✅ `.json`
- ✅ `.jsonl`
- ✅ `.txt`
- ✅ `.parquet`

### 🛠️支持自定义类型，请按照下面方法补充代码
1. 进入[MetaFileReadTool.py](../tools/MetaFileReadTool.py)
2. 在METADATA_READERS上方创建自定义读取方法，返回值应该是一个字典，其中包含两个键值对：type和data，前者可选为raw_sample（可直接输入大模型的模态如文本的txt等）或structured（复杂模态如图像、音频等）或structured_summary（更复杂的如多模态数据集），后者包含提供给大模型用于理解数据集的必要信息（如图像的长宽通道数、表格的题头等）
3. 在METADATA_READERS中添加类型后缀和读取方法名的映射
4. 保存、运行✈️
