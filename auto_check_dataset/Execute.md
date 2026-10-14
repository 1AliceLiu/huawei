> Name:auto check dataset execute 
Discription: 这个文档用于执行数据的自动化审查步骤

# Step

- 选择要筛选的**已标注**的数据
- 执行 main.py 将标注好的图片的**ROI**裁剪下来放在**crop文件夹**中
- 随后对ROI进行特征提取，最后对特征进行聚类以及分析
- 最后得到每个类别前top10高质量、离群的ROI（find_dirty.py）等。
- 我们将那些质量较差的ROI进行分类，分成：模糊的、过小的、离散的、离群的、mislabel、重复的...（build_suspect_sets.py）存放至**suspect_select**文件夹里，images和label是以原图片的形式呈现，**items才是这个类别的尚未修改的框的相关信息**。
- 然后我们根据这些分类建立了一个可以在本地运行html，我们在上面可以删除、保留框以及更改类别，更改后的结果将保存在clean_data文件夹中，labels也会进行相应的更改。**每个框都有可能同时出现在不同的类别里，labels是在修改后的基础上再进行修改，不是覆盖。**
- **这里建议首先将duplicate类别的框进行一个筛选，去除掉所有场景、类别近似的图片，这样能省下不少人工成本。注意在筛选时的规则应该是：尽量保持多种类别同时要保留Van/motorcycle/LGV/Contatiner/Coach这种稀有类别。**