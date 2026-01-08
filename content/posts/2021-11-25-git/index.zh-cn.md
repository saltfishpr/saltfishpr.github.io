---
title: Git 常用命令
date: 2021-11-25T00:49:31+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
tags: ["git"]
categories: ["Cheat Sheet"]
---

<!--more-->

## 初次运行 Git 前的配置

设置用户名和邮箱

```shell
git config --global user.name "Salt Fish"
git config --global user.email saltfishpr@gmail.com
```

配置默认文本编辑器

```shell
git config --global core.editor code
```

检查配置信息

```shell
git config --list
```

## 设置保存密码

```shell
# 记住密码（默认 15 分钟）
git config --global credential.helper cache

# 自己设置时间
git config --global credential.helper cache --timeout=3600

# 长期存储密码
git config --global credential.helper store
```

增加远程地址的时候带上密码

```shell
git remote -v

# https://yourname:password@github.com/saltfishpr/go-learning.git
```

## Git clone/push 太慢

在国内，github 域名被限制，导致 git clone 很慢，只有 40KB/s 的速度

### Windows

使用 `nslookup` 查询 github 对应的 IP 地址

```shell
nslookup github.global.ssl.fastly.net
nslookup github.com
```

把查询到的结果添加到 `C:\Windows\System32\drivers\etc\hosts` 中

```text
github.global.ssl.fastly.net 69.63.184.14
github.com 140.82.112.3
```

刷新 DNS 缓存

```shell
ipconfig /flushdns
```

### linux

使用 `nslookup` 查询 github 对应的 IP 地址

```shell
nslookup github.global.ssl.fastly.net
nslookup github.com
```

把查询到的结果添加到 `/etc/hosts` 中

```text
github.global.ssl.fastly.net 69.63.184.14
github.com 140.82.112.3
```

刷新 DNS 缓存

```shell
sudo nscd -i hosts
```

## 清除远程分支的本地缓存

当修改/删除远程分支后，本地的远程分支缓存未被删除，再 checkout 会出现 `remote ref does not exist` 错误，这时要先清除本地缓存

```shell
git fetch -p origin
```

## rebase

将当前分支变基到目标分支

```shell
git rebase [target] [current]
```

常在 push 代码前使用，如下将个人分支变基到远端开发分支，本地处理冲突后提交

```shell
git rebase origin/dev dev_xxx
```

## cherry pick

- 将某一 (几) 次提交复制到当前分支

  ```shell
  git cherry-pick <commit id A> <commit id B>
  ```

- 将某一段提交复制到当前分支

  ```shell
  git cherry-pick A..B # 不包含 A
  git cherry-pick A^..B # 包含 A
  ```
