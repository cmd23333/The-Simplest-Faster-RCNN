# Simplest-Faster-RCNN
The simplest Faster-RCNN with only one page of jupyter notebook.

(More Detials to be append... & maybe a blog would be written.)

Drop any additional improvements and tricks based on simple-faster-rcnn(https://github.com/chenyuntc/simple-faster-rcnn-pytorch).

and Regroup the pipeline according to the order of how faster rcnn runs.

## New Datasets~
League of Legends' champions were labeled. (Actually only Fiona, Jinx, Zoe, Irelia, Akali, Ashe, Evelynn were used. ^-^ as I love them.)
The file The-Simplest-Faster-RCNN-demo.ipynb was modified to suit our datasets. here is a result：
![result](https://github.com/cmd23333/The-Simplest-Faster-RCNN/blob/master/imgs/result.png)

## Details:

    VGG16 for backbone

    SGD only
    
    No data augument like flip or crop
    
    No "visdom" visualization
    
    No six (6 = 2*3), means the program only support for Python3.x
    
    Add many comments on nearby statements
     
## the notebook has the 9 main parts as the pic show below.

1.Library<br>
2.Config<br>
3.Dataset<br>
4.Backbone<br>
5.RPN<br>
6.ROI<br>
7.FasterRCNN<br>
8.Trainer<br>
9.Training<br>

![notebook](https://github.com/cmd23333/The-Simplest-Faster-RCNN/blob/master/imgs/fileLook.png)


Maybe you need to download some extension of jupyter notebook like Codefolding (See below.)
![Extention](https://images.gitee.com/uploads/images/2020/0325/211646_78e5f010_5361668.png)
## requirements:

### librarys:(see Section1 of the main file.)

    torch
    
    PIL
    
    torchvision
    
    torchnet
    
    cupy

### Dataset:

    VOC2007

## You may find the dataset and the pre-train model in simple-faster-rcnn.

## Faster RCNN的训练流程
我们假设只有一张图的情况。比如我们输入图片的大小是600×600，图片上有两个待检测的目标。那么整个Faster RCNN的输入就是这张图片，形状是1×600×600×3，因为是彩色图片，有RGB三个通道；还有就是检测目标的信息，包括了位置信息和类别信息，分别用bbox和label表示，二者的形状是1×2×4和1×2。这边第一个维度的1代表batch size。然后Faster RCNN算法支持任意大小的图片输入的，所以一般还有一个scale信息，代表这张图片被放缩了多少。这是为了让训练过程比较稳定，原始论文中的算法会将训练集中的图片按照长边不超过1000，短边不超过600这样缩放到半固定的尺寸，当然图片中待检测的物体坐标也对应的放缩。
我们先不考虑这个的话，这张图片进入到Faster RCNN，最先经过一个backbone网络提取特征，也就是feature map。原始的算法使用了VGG16作为主干网络。VGG16里有4个2×2的max pooling层，其余卷积层都是SAME形式，不改变长宽这两个维度。所以经过backbone之后，我们获得了这张图片的feature map，形状为512×37×37，其中37是600在池化过程中缩小16倍并且向下取整的结果，512的feature map的通道数。
得到feature map之后，我们进入RPN网络，RPN是2-stage网络的第一个1-stage，用来获得图片中比较有可能有目标的区域，并且送到后续的网络中做进一步的计算。具体来说的话，我们将feature map送入RPN网络，首先先用一个3×3的卷积接着RELU激活函数对feature map做进一步的特征提取，还是512个通道，得到新的特征图，形状仍然是512×37×37，接下来将这个新的特征图分别传给两条支路，其中一条是36个1×1卷积核的卷积层，后面不加激活函数，可以得到36×37×37的输出，代表了可能有目标的区域的位置信息；另一条支路是18个1×1的卷积核，得到18×37×37的输出，代表这些区域有目标和没有目标的概率。然后这个36和18是怎么来的呢。这就是算法在做目标检测的时候，给特征图上的每个点都生成了9个anchors（三种不同尺寸、三种不同长宽比），anchors相当于一个先验信息，我们想要知道这些anchors内是否包含物体，以及物体的坐标。而36就代表着9×4，每个anchors都有其对预测目标的坐标偏移量，注意这里并不是直接给出每种anchor预测物体的x1,y1,x2,y2这样的绝对坐标，而是计算了x,y,w,h的偏移量dx,dy,dw,dh，后面是通过，比如预测物体的中心坐标x等于anchor的中心坐标Ax加上anchor的宽度Aw乘以我们网络计算的偏移量dx得到，同理，预测物体的高度等于Ah × exp(dh)。而另一个支路，也就是18×37×37的前景后景得分，我们会先把他reshape成2×9×37×37并且在2的那个维度做一次softmax，可以保证得到的前景后景概率大于0且二者之和是1。
接下来，前面生成了37×37×9约15000个位置信息和置信度信息，我们需要从中选择一些在传给后面的网络。具体来说，首先，要先将约15000个dx,dy,dh,dw根据每个anchor的坐标转换成预测区域在原图的x1,x2,y1,y2坐标。进行clip操作，让roi的坐标不能超出原图的边界，然后根据前景概率得分，选择分数靠前的约12000个roi，然后使用nms去重，保留至多2000个roi，将这2000个roi的坐标（x1,x2,y1,y2）传递给下一个stage。
然后计算损失函数的时候，我们不会算15000个，也不是算传递给下一阶段的2000个，而是从这约15000个roi中选取256个。具体来说，我们计算每个ground turth和预测roi的IoU，对于每个gt，我们先挑出和它IoU最大的2个（因为我们假设图片里就两个目标）作为正样本，再从IoU大于0.7的roi中挑选最多126个，如果没有这么多也没关系。剩下的，我们要从IoU小于0.3的roi中补足256个当作负样本。
当然在训练过程中我们并不是把这2000个roi都送入下一阶段，我们只是从其中挑选出128个进行训练。具体来说，这128个roi分别是：和ground turth的IoU大于0.5的，我们认为是正样本，选择最多32个；然后和gt的IoU介于0.1-0.5之间的，我们补齐128个。这样我们得到了roi head网络的输入是128×4的rois，和前面得到的512×37×37的feature map一同作为输入。首先是roi pooling层，在选择roi在feature map对应的区域，做roi pooling操作得到7×7的特征，输出一个128×512×7×7的特征向量。
注意到VGG16最后一个卷积后得到的向量也是batch size,512×7×7的，后续是将其flatten在送到全连接层做分类。因此我们也把128×512×7×7的向量reshape成128×25088，然后就可以作相当于是一个迁移学习，经过两个全连接层得到128×4096的向量（这两个全连接层的权重可以用VGG16的权重）。后面再用两个全连接网络分别得到128×21和128×84的分类信息和位置信息。
后面再计算损失函数，做一次反向传播，Faster RCNN的训练流程基本就这样了。

学习经验：观察每一步变量的shape，在PyCharm或其他工具中使用Debug模式逐条语句运行，可以帮助快速的了解一个框架！

## Also, here is a hand-writing notebook 
   and it's about how a batch of data (in this case, 1 img) goes through the training step,with all the shape of intermediate variables are marked.
   
   Forgive my Scribbled handwriting.

![notes](https://github.com/cmd23333/The-Simplest-Faster-RCNN/blob/master/imgs/notesLook.png)

## Ackownledgement:
simple-faster-rcnn (https://github.com/chenyuntc/simple-faster-rcnn-pytorch)<br>
blogs(https://www.cnblogs.com/king-lps/category/1208643.html)<br>
