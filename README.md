# Simplest-Faster-RCNN
The simplest Faster-RCNN with only one page of jupyter notebook.

(More Detials to be append... & maybe a blog would be written.)

Drop any additional improvements and tricks based on simple-faster-rcnn(https://github.com/chenyuntc/simple-faster-rcnn-pytorch).

and Regroup the pipeline according to the order of how faster rcnn runs.

## Details:

    VGG16 for backbone

    SGD only
    
    No data augument like flip or crop
    
    No "visdom" visualization
    
    No six (6 = 2*3), means the program only support for Python3.x
    
    Add many comments on nearby statements
     
## the notebook has the 10 main parts as the pic show below.

1.Library<br>
2.Config<br>
3.Dataset<br>
4.Backbone<br>
5.RPN<br>
6.ROI<br>
7.FasterRCNN<br>
8.Trainer<br>
9.Training<br>
10.Evaluating<br>

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

## Also, here is a hand-writing notebook about how a batch of data (in this case, 1 img) goes through the training step,
   with all the shape of intermediate variables are marked.
   Forgive my Scribbled handwriting.

![notes](https://github.com/cmd23333/The-Simplest-Faster-RCNN/blob/master/imgs/notesLook.png)

## Ackownledgement:
simple-faster-rcnn (https://github.com/chenyuntc/simple-faster-rcnn-pytorch)<br>
blogs(https://www.cnblogs.com/king-lps/category/1208643.html)<br>
