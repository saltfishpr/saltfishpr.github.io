---
title: Python 环境管理
date: 2020-03-26T00:00:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
tags: ["python"]
categories: ["Programming"]
---

<!--more-->

## conda

### 换源

创建 .condarc 配置文件

```bash
conda config --set show_channel_urls yes
```

用文本编辑器打开 `~/.condarc` 填入以下内容

```
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

### 命令

创建有最新版本的 python 的环境 `conda create -n <env-name> python=3`

删除环境 `conda remove -n <env-name> --all`

清除缓存 `conda clean -a`

## pip

### 换源

在 `~/.config/pip/pip.conf` 中添加如下内容

```
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host=mirrors.aliyun.com
```

### 删除缓存

找到 `~/.cache/pip` 文件夹，删除即可

在安装时使用

```bash
pip install <package-name> --no-cache-dir
```

## 导入导出 python 环境

今天在学习数据挖掘的时候，nolearn 和 lasagne 两个库的时候给我的 jypyterlab 环境搞崩了，只好 remove --all 重新配起，真后悔没有先搞个环境备份 ( ´•︵•` )

### conda

conda 是个好东西，可以自己处理环境依赖，缺点就是。。包有点老，有些包还找不到

导入环境

```bash
conda create --name <your env name> --file <this file> --yes
```

导出环境

```bash
conda list -e > requirements.txt
```

### pip

pip 也有很多优点，我一般是在 conda 找不到模块的时候使用 pip

导入环境

```bash
pip install -r requirements.txt
```

导出环境

```bash
pip freeze > requirements.txt
```
