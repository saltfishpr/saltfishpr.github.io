---
title: Github Pages 个人博客搭建
date: 2020-03-10T00:00:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
tags: ["jekyll"]
categories: ["Skill"]
featuredImage: cover.jpg
---

<!--more-->

应朋友的强烈要求在这里记录下使用 `github pages` + `jekyll` 搭建个人网站的步骤。我自己花了两天梳理了一下 jekyll 模板的目录结构和使用方法，只会些简单的修改但是基本够用了（毕竟不是前端人）。

## 准备

- 注册 github 账号
- 创建一个仓库，命名必须为&lt;username&gt;.github.io，百度上任意搜索都有介绍过程，这里就不再赘述了

### windows:

- 安装 [git](https://git-scm.com/)，下载安装最新版本即可

- 安装 [msys2](https://mirrors.tuna.tsinghua.edu.cn/msys2/distrib/msys2-x86_64-latest.exe "Msys2 清华镜像")

  安装好 msys2 后，首先记得切换[镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/msys2/ "msys2镜像源设置")，不然会因为网速太慢心态崩溃。在设置镜像源的时候，可以将除了清华源的其他源全部注释掉，提高下载速度。

- 安装 [ruby](https://github.com/oneclick/rubyinstaller2/releases/download/RubyInstaller-2.6.5-1/rubyinstaller-devkit-2.6.5-1-x64.exe "ruby+devkit 2.6.5")，这里下载 2.6.5 版本，因为最新版后面会提示版本不匹配。

  - 下载过程可能很久...如果觉得慢可以下载[不包含 devkit 的版本](https://rubyinstaller.org/downloads/)。
  - 安装时在 Select Components 界面不用勾选 msys2，因为在上一步中已经安装好了。
  - 在 Finish 时勾选 Run 'ridk install' to setup msys2...
  - 弹出配置界面，在这里我选择 3 并按回车。由于配置过 msys2 的源，这里下载安装速度很快。
  - 安装完成后，打开 msys2 命令行窗口，为 rubygems[配置源](https://mirrors.tuna.tsinghua.edu.cn/help/rubygems/)。

- 安装[rubygems](https://rubygems.org/rubygems/rubygems-3.1.2.zip)，下载后解压缩，在文件夹中打开 git bash 输入 `ruby setup.rb`

- 安装 bundler：在 msys2 或者 git bash 命令行中输入`gem install bundler`，等待安装完成

  - 再次使用[清华源](https://mirrors.tuna.tsinghua.edu.cn/help/rubygems/)为 bundle 配置镜像源。

- 安装 jekyll：在命令行中输入`gem install jekyll`

至此准备工作全部完成～

### linux(Debian 10.3)

debian 系统自带大部分基础包

- git: `sudo apt-get install git`
- 为 gem 配置镜像源，步骤与 Windows 一样
- 安装 bundler: `gem install bundler`
- 为 bundler 配置镜像源，同上
- 安装 jekyll: `gem install jekyll`

至此准备工作全部完成～

## 搭建 blog

使用 git clone 将自己的仓库 clone 下来。

如果有能力可以自己生成一个新的 jekyll 项目：新建一个空文件夹，在此文件夹中命令行输入`jekyll new site-name`，将这个文件夹的所有文件复制到 clone 下来的仓库中。

也可以在[模板网站](http://jekyllthemes.org/)下载一个修改配置。我的[github page](https://github.com/saltfishpr/saltfishpr.github.io)使用的是 [Hux](http://huangxuan.me/) 大佬的模板。

这里可以克隆我的模板  
`git clone https://github.com/saltfishpr/saltfishpr.github.io.git`  
在项目文件夹中打开 git bash 输入`bundle install`，等待安装完成。如果安装出现依赖问题，可以在百度中搜索缺少的包看如何安装，都可以找到解决答案。

项目文件的编辑推荐使用 vscode。用 vscode 打开文件夹后，可以看到文件夹的目录结构：

    ├── 404.html
    ├── about.html
    ├── archive.html
    ├── CNAME
    ├── _config.yml
    ├── css
    ├── feed.xml
    ├── fonts
    ├── Gemfile
    ├── Gemfile.lock
    ├── Gruntfile.js
    ├── img
    ├── _includes
    ├── index.html
    ├── js
    ├── _layouts
    ├── less
    ├── LICENSE
    ├── offline.html
    ├── _posts
    ├── pwa
    ├── README.md
    ├── _site
    └── sw.js

这里主要修改配置文件\_config.yml。

## 运行本地服务

在项目根目录命令行中输入`bundle exec jekyll server --watch`，在本地启动 jekyll 服务器，浏览器输入 127.0.0.1:4000 本地查看 blog。

对照\_config.yml 修改；善用 ctrl+shift+F 全局搜索。修改出属于自己的 blog 吧！

## 写在最后

由于建立 blog 是在一个星期前了，可能有些小步骤记不清楚，有什么问题欢迎下方 issue 指正 O(∩_∩)O。

## issues

缺少 ruby.in : `sudo apt-get install ruby-dev`

zlib is missing; necessary for building libxml2 : `sudo apt-get install zlib1g zlib1g.dev`

参考资料：

1. [jekyll 目录结构](http://jekyllcn.com/docs/structure/)
2. [github pages 文档](https://help.github.com/en/github/working-with-github-pages/adding-a-theme-to-your-github-pages-site-with-the-theme-chooser)
