---
layout: post
title: "教你如何一句代码看不懂也能写一个自己的blog主页出来"
subtitle: 'Build your own blog on github.io'
author: "Jiaqing Zhang"
header-style: text
tags:
  - Blog
  - 干货
---

废话不多讲，直接开干货：

## Step 1
首先注册一个github账号，然后新建一个repo，命名为`<username>.github.io`，其中，`<username>`**是你github账号名**，例如我的blog地址为`danielqingz.github.io`。

![img](/img/in-post/post-210621-1.jpg)

## Step 2
向你空的repo中添加模版文件，本例子在[Hux Blog](https://github.com/Huxpro/huxpro.github.io)的基础上进行了简化和修改，你可以直接参考该目录，制作你自己的blog。

对于熟悉`git`的同学，你当然可以将自己的目录`clone`到本地，然后`fork`上述的模版，进行编辑。

如果不熟悉`git`操作也没有关系，你可以直接将Hux Blog中的代码统统下载到本地，然后在本地进行修改后，直接通过拖拽上传上传到你自己的repo目录下。

## Step 3
需要修改的内容较多，主要有：
- `_config.yml`
- `img/`目录下的图片内容
- `_posts/`目录下的blog文本内容

`_config.yml`是重点修改对象，网页的名字、社交网站链接、个人信息等基本内容在本文件内即可完成修改

`img/`目录下的图片，你可以按需替换成自己的内容，网页的背景图片，你可以直接将原有的图片删除，并上传自己喜欢的图片，修改为跟之前相同的名字即可，其中`in-post/`目录下存放在blog内容所引用的图片

`_posts/`目录用于存放你的blog内容，内容格式可参考`_posts/2021-06-21-build-your-own-blog.markdown`的格式内容，直接修改为自己的内容即可，文件的命名格式必须为`yyyy-mm-dd-your-title.markdown`

## Step 4
将修改好的文件，拖拽上传到你先前建好的repo中，commit后，就大功告成了！

打开你的`<username>.github.io`，就可以查看你Blog的内容了！

## 额外的Step 5
本文只介绍了建立blog最基本的操作，如果你对需要进行更为进阶的操作，可以学习[Jekyll](https://jekyllrb.com/)；另外，本repo有很多有趣的较为高级的功能，如搜索功能、提示网页刷新功能、离线阅读功能、自动生成词云功能、增加外挂评论区功能等，均保留在代码中，感兴趣的同学可以进一步探索。

# Reference

[Github Page Doc](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)

[Hux Blog](https://github.com/Huxpro/huxpro.github.io) 

