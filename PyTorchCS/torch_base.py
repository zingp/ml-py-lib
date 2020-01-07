{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "# import torchvision\n",
    "from torch.autograd import Variable\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0.1  |  0.2.1\n"
     ]
    }
   ],
   "source": [
    "print(torch.__version__, \" | \",torchvision.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[0.5150, 0.1147, 0.7260, 0.6619, 0.2875],\n",
       "        [0.2622, 0.6950, 0.4466, 0.2400, 0.7938],\n",
       "        [0.7241, 0.3063, 0.0523, 0.1012, 0.5750]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t = torch.rand(3,5)   # 创建随机张量，数字0-1之间\n",
    "t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.float32"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t.dtype  # 默认是torch.float32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.is_tensor(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.is_storage(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "?torch.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[-1.4732,  0.9982, -1.0748,  0.2213, -1.6524],\n",
       "        [ 0.7433, -1.0574,  1.2384, -1.0754,  0.2201],\n",
       "        [ 0.6080,  0.1292,  0.7955,  0.3233,  0.5500]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.randn(3, 5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[0., 0., 0.],\n",
       "        [0., 0., 0.]])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.zeros(2, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[1., 1., 1.],\n",
       "        [1., 1., 1.]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.ones(2, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.float32"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.float    # float32 \n",
    "torch.float16  # float16\n",
    "torch.float32  #\n",
    "torch.float64\n",
    "\n",
    "torch.int    # int32\n",
    "torch.long   # int64\n",
    "torch.int8   # int8\n",
    "torch.int16\n",
    "torch.int32\n",
    "torch.int64"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "torch.set_default_dtype(torch.float64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.float64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.tensor([1.0002345, 2.39087]).dtype  # torch.tensor默认是float32，注意设置default之后的改变"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Tensor 与Numpy 转换"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([[1,2], [3,4]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [3, 4]])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "a_torch = torch.from_numpy(a)  # numpy to tensor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[1, 2],\n",
       "        [3, 4]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a_torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "a_np = a_torch.numpy()    # tensor to numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [3, 4]])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a_np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Suport GPU  CPU  to trans\n",
    "- 首先通过 torch.cuda.is_available() 判断一下是否支持GPU，如果想把tensor a放到GPU上，只需 a.cuda()就能够将tensora放到GPU上了。\n",
    "```python\n",
    "if torch.cuda.is_available(): \n",
    "    a_cuda = a.cuda()\n",
    "    print(a_cuda)\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[1, 2],\n",
      "        [3, 4]], device='cuda:0')\n"
     ]
    }
   ],
   "source": [
    "if torch.cuda.is_available(): \n",
    "    a_cuda = a_torch.cuda()\n",
    "    print(a_cuda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[1, 2],\n",
       "        [3, 4]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a_cuda.cpu()      # cuda tensor  to cpu tensor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [3, 4]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a_cuda.cpu().numpy()   #  cuda tensor  to cpu numpy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Variable\n",
    "- torch.autograd.Variable中，要将一个tensor变成Variable也非常简单，比如想让一个tensora变成Variable，只需要Variable(a)就可以了。\n",
    "- Variable 有三个比较重要的组成属性: item() , grad 和 grad_fn 通过 item() 可以取出 Variable 里面的 tensor 数值， grad_fn 表示的是得到这个Variable的操作，比如通过加减还是乘除来得到的，最后 grad 是这个Variable 的反向传播梯度.下面通过例子来具体说明一下:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = Variable(torch.Tensor([1]) , requires_grad=True)  # requires_grad=True，这个参数表示是否对这个变量求梯度，默认的是False，也就是不对这个变量求梯度\n",
    "w = Variable(torch.Tensor([2]) , requires_grad=True)\n",
    "b = Variable(torch.Tensor([3]) , requires_grad=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = w*x + b   # y = 2x + 3   计算图"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "y.backward()   # 计算梯度"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([2.])\n"
     ]
    }
   ],
   "source": [
    "print(x.grad)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0\n"
     ]
    }
   ],
   "source": [
    "print(x.item())   # 得到变量的数值"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([1.])\n"
     ]
    }
   ],
   "source": [
    "print(w.grad)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([1.])\n"
     ]
    }
   ],
   "source": [
    "print(b.grad)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "None\n"
     ]
    }
   ],
   "source": [
    "print(x.grad_fn)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 标量求导和变量求导"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([-0.7025,  1.3281, -2.8397])\n"
     ]
    }
   ],
   "source": [
    "x = torch.randn(3)\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = Variable(x, requires_grad=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = 2*x + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "y.backward(torch.FloatTensor([1, 0.1, 0.5]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([2.0000, 0.2000, 1.0000])\n"
     ]
    }
   ],
   "source": [
    "print(x.grad)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset\n",
    "- 在处理任何机器学习问题之前都需要数据读取，并进行预处理。 PyTorch提供了很多工具使得数据的读取和预处理变得很容易。\n",
    "- torch.utils.data.Dataset是代表这一数据的抽象类，你可以自己定义你的数据类继承和重写这个抽象类，非常简单，只需要定义__len__和__getitem__这两个函数:\n",
    "```python\n",
    "class myDataset(Dataset) :\n",
    "    def init (self, csv_file, txt_file, root_dir, other_file):\n",
    "        self.csv_data = pd.read_csv(csv_file ) \n",
    "        with open(txt_file, 'r' ) as f:\n",
    "            data_list = f.readlines()\n",
    "        self.text_data = data_list\n",
    "        self.root_dir = root_dir\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.csv_data)\n",
    "    \n",
    "    def __getitem__(self):\n",
    "        data = (self.csv_data[idx], self.txt_data[idx])\n",
    "        return data\n",
    "```\n",
    "- 通过上面的方式，可以定义我们需要的数据类，可以通过迭代的方式来取得每一个数据，但是这样很难实现取batch， shuff1e 或者是多线程去读取数据，所以PyTorch中提供了一个简单的办法来做这个事情，通过torch.utils.data.DataLoader 来定义一个新的迭代器。如下:\n",
    "```python\n",
    "dataiter = DataLoader(myDataset, batch_size=32, shuffle=True, collate_fn=default_collate)\n",
    "```\n",
    "- 里面的参数都特别清楚，只有 collate_fn 是表示如何取样本的，我们可以定义自己的函数来准确地实现想要的功能，默认的函数在一般情况下都是可以使用的。\n",
    "```python\n",
    "dset = ImageFolder(root='root-path', transform=None, loader=default_loader)\n",
    "```\n",
    "- 其中的 root 需要是根目录，在这个目录下有几个文件夹，每个文件夹表示一个类 transform 和 target_transform 是图片增强，之后我们会详细讲; loader 是图片读取的办法，因为我们读取的是图片的名字， 然后通过 loader 将图片转换成我们需要的图片类型进入神经网络。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## nn.Module\n",
    "- PyTorch里面编写神经网络，所有的层结构和损失函数都来自于 torch.nn，所有的模型构建都是从这个基类 nn.Module 继承的，于是有了下面这个模板。\n",
    "```python\n",
    "class net_name(nn.Module):\n",
    "    def __init__(self, other_arguments):\n",
    "        super(net_name, self).__init__()\n",
    "        self.conv1 = nn.Conv2d()\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = self.conv1(x)\n",
    "        return x\n",
    "```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 模型的加载与保存\n",
    "- 模型的保存和加载在 PyTorch里面使用 torch.save来保存模型的结构和参数，有两种保存方式:\n",
    "    (1)保存整个模型的结构信息和参数信息，保存的对象是模型model; \n",
    "    (2)保存模型的参数，保存的对象是模型的状态 model.state dict()。 \n",
    "- 可以这样保存，torch.save()的第一个参数是保存对象，第二个参数是保存路径及名称:\n",
    "```python\n",
    "torch.save(model, './model.pth ' ) \n",
    "torch.save(model.state_dict(), './model_state.pth')\n",
    "```\n",
    "- 对应的加载模型有两种方式:\n",
    "    (1)加载完整的模型结构和参数信息，使用 load_model = torch.load('model.pth') ，在网络较大的时候加载的时间比较长，同时存储空间也比较大;\n",
    "    (2)加载模型参数信息，需要先导人模型的结构，然后通过 model.load_state_dic(torch.load('model_state.pth')) 来导入。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  Tensor的一些方法"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = torch.FloatTensor([\n",
    "    [[[1,2], [2,3], [3,4]]],\n",
    "    [[[1,2], [2,3], [3,4]]]\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 1, 3, 2])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[1., 2., 2., 3., 3., 4.],\n",
       "        [1., 2., 2., 3., 3., 4.]])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.view(x.shape[0], -1)   # 或者x.view(x.size(0), -1)  压平成一个二维，且第一个维度是x.size(0)。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
