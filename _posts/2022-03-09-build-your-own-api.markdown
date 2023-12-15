---
layout: post
title: "服务器上docker+uwsgi+flask部署API"
subtitle: 'Build your own Python API on server'
author: "Jiaqing Zhang"
header-style: text
tags:
  - Docker
  - API
  - 干货
---

废话不多讲，直接开干货：

## Step 0
环境：CentOS（腾讯云）

## Step 1
在服务器上安装docker环境：
```
yum install -y yum-utils

# 安装docker-ce社区版（docker-ee是企业版）
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
yum install -y docker-ce docker-ce-cli  containerd.io --nobest

# 查看版本，确认docker安装完成
docker -v

# 启动docker
systemctl start docker

```

P.S.：

可能报错：`Cannot connect to the Docker daemon at unix:/var/run/docker.sock. Is the docker daemon running?`。

解决办法：没打开docker，`systemctl start docker`，打开就行。

## Step 2


<!-- ![img](/img/in-post/post-210621-1.jpg) -->

TBD!


# Reference

[Github Page Doc](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)

[Hux Blog](https://github.com/Huxpro/huxpro.github.io) 

