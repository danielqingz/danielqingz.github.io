---
layout: post
title: "教你如何一句代码看不懂也能写一个自己的blog主页出来"
subtitle: 'Build your own blog on github.io'
author: "Jiaqing Zhang"
header-style: text
tags:
  - Blog
  - JS
---

## Step 1
首先注册一个github账号，然后新建一个repo，命名为`<username>.github.io`，其中，`<username>`**是你github账号名**，例如我的blog地址为`danielqingz.github.io`。

## Step 2
向你空的repo中添加模版文件，本例子在[Hux Blog](https://github.com/Huxpro/huxpro.github.io)的基础上进行了简化和修改，你可以直接参考[Jiaqing Zhang's Blog](https://github.com/danielqingz/danielqingz.github.io)制作你自己的blog。

对于熟悉`git`的同学，你当然可以将自己的目录`clone`到本地，然后`fork`上述的模版，进行编辑。

如果不熟悉`git`操作也没有关系，你可以直接将Jiaqing Zhang's Blog中的代码统统下载到本地，然后在本地进行修改后，直接通过拖拽上传上传到你自己的repo目录下。

## Step 3
需要修改的内容较多，主要有：
- `_config.yml`
- `img/`目录下的图片内容
- `_posts/`目录下的blog文本内容

`_config.yml`是重点修改对象，网页的名字、社交网站链接、个人信息等基本内容在本文件内即可完成修改

`img/`目录下的图片，你可以按需替换成自己的内容

`_posts/`目录用于存放你的blog内容，内容格式可参考本文的`markdown`格式内容，直接修改为自己的内容即可

## Step 4
将修改好的文件，拖拽上传到你先前建好的repo中，commit后，就大功告成了！
