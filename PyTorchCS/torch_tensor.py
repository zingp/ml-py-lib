#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# # torch tensor常用操作

# In[3]:


x = torch.tensor([[1,2,3,4],
                  [4,3,2,1]], 
                 dtype=torch.float32)
x


# ## x.argmax(dim=n) 
# - 按照维度，取最大值的索引

# In[4]:


x.argmax(dim=1)  # 按照列，实际上就是行


# ## x.softmax(dim=n)
# - 相加得1的概率分布

# In[5]:


x.softmax(dim=1)


# In[6]:


y = torch.randn(1,4)
y


# ## torch.cat((x, y), dim=0)
# - 对数据沿着某一维度进行拼接。
# - cat后数据的总维数不变。
# - 比如下面代码对两个2维tensor（分别为2*4,1*4）进行拼接，拼接完后变为3*4还是2维的tensor。

# In[7]:


torch.cat((x, y), dim=0)  # 合并，其余维度相同


# ## torch.stack() 增加新的维度进行堆叠
# - stack则会增加新的维度。
# - 如对两个1*2维的tensor在第0个维度上stack，则会变为2*1*2的tensor；在第1个维度上stack，则会变为1*2*2的tensor。

# In[8]:


a = torch.randn(1,2)
a


# In[9]:


b = torch.randn(1,2)
b


# In[10]:


c = torch.stack((a, b), dim=0)
c


# In[11]:


c.size()


# In[12]:


d = torch.stack((a, b), dim=1)
d


# ## transpose 交换维度
# - 维度互换，只能两个维度

# In[13]:


x


# In[14]:


x.transpose(0,1)


# In[15]:


x.transpose(1,0)


# In[16]:


x = torch.randn(2,3,5)
x


# In[17]:


x.transpose(1,2)


# In[18]:


x.transpose(1,2).size()


# ## x.permute()
# - 适合多维数据，更灵活的transpose
# - permute是更灵活的transpose，可以灵活的对原数据的维度进行调换，而数据本身不变。

# In[19]:


x = torch.randn(2,3,5)
x


# In[20]:


y = x.permute(1,2,0)
y.size()


# In[21]:


y


# ## x.reshape()
# - 数据不变，改变tensor的形状

# In[22]:


x.shape


# In[23]:


x.reshape(3,1,10)


# In[24]:


x.reshape(5,6)


# ## x.view()
# - 改变形状

# In[25]:


x.view(-1, 2, 3)


# In[26]:


x.view(-1, 15)


# In[27]:


x


# ## x.unsqueeze/squeeze
# - squeeze(dim_n)压缩，即去掉元素数量为1的dim_n维度。
# - 同理unsqueeze(dim_n)，增加dim_n维度，元素数量为1。

# In[28]:


x = torch.randn(5,1,3)
x.size()


# In[29]:


x.squeeze(dim=1).size()


# In[30]:


# squeeze 挤压
x.unsqueeze(dim=0).size()


# In[ ]:




