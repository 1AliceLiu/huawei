---
name:json-to-yolotxt
description:应用于将人工标注好的json文件转变为yolo26适用的labeltxt文件
---

# begain
将json文件转变为yolo26适用的labeltxt文件时,遵循一下规范:
## step
-人工标注好的json文件存放在C:\Users\admin\Desktop\projects\huawei\tmp中
-首先列出tmp文件夹中所有的子目录并询问我:需要将哪个子目录中的所有json文件进行转变
-等我给出答案后,进入对应的子目录,随后进入images文件夹
-此时遍历该目录中每一个json文件,提取所有rectangle shape的"label"和"points"
-将每一个rectangle shape的label对应../classes.txt中的顺序,从0开始标记
-所有rectangle shape的points表示框的四个角的坐标,而我需要的是计算出框中心点在整幅图的相对位置
-最后的格式参照dataset\batch_12.v14i.merged12.yolov11\valid\labels\23_jpg.rf.6c3fd38d98b1ca77fbd5a74fa93c0b92.txt每一行代表一个rectangle shape
-最后为了验证label的txt文件是否正确,我需要你在tmp\r10_pending文件夹中生成一张图片:根据tmp\r10_pending\images\20261005_150604_341800_dahua1001619.json生成的labeltxt文件的类别和框坐标,在tmp\r10_pending\images\20261005_150604_341800_dahua1001619.jpg上画出所有框并标出类别.
-先询问我是否通过了验证,如果通过验证你再将所有txt文件保存至tmp\r10_pending\label文件夹中,命名依旧是图片的名字,只是后缀换成txt