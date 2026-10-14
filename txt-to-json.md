> Name:Label dataset txt format to json

## step
- 首先读取suspect_select文件夹下的目录，询问我想要处理哪一类数据
- 读取相应文件夹下的labels文件夹中每一个txt文件
- labels文件夹中每一个txt文件都是一一对应着images文件夹的
- 参照20260917_131612_545126_cam2181.json文件中的信息，你只需要将构建一个shape，将label的位置对应label文件的第一列数字类别，其中数字类别对应着dataset/Dahua2/classes.txt文件
- points对应着txt文件的后四列，你需要读取image的大小并将txt文件框中心坐标转换为框的四个角的坐标
- "attributes": {
        "conf": 0.9018
      }不需要写上去
- 以此类推，将一张图中所有的框写在json文件中
- 注意json文件的命名需要与image一一对应
- 你先将一张图片的label txt文件转为对应的json文件并将image和json文件一起保存至根目录中供我验证
- 经过我的验证后，将根目录供我验证的image和json文件删除
- 开始全量转换并将所有的json文件存放至images文件夹中
- 最后检查所有images文件夹中所有jpg文件和json文件名字数量是否一致