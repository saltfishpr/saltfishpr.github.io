---
title: Markdown 常用写法大全
date: 2020-03-05T00:00:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
tags: ["markdown"]
categories: ["Cheat Sheet"]
---

<!--more-->

## 分割线

```markdown
---
```

## 标题

```markdown
# 一级标题

## 二级标题

### 三级标题

###### 六级标题
```

## 斜体文字

```markdown
_斜体_
```

_斜体_

## 加粗文字

```markdown
**加粗**
```

**加粗**

## 删除线

```markdown
~~删除内容~~
```

~~删除内容~~

## 段落和换行

```markdown
新一段落需要空一行

新的一段

或者在最后加上两个空格  
换行
```

新一段落需要空一行

新的一段

或者在最后加上两个空格  
换行

## 列表

### 基本用法

```markdown
- 列表
- 列表
- 列表

1. 列表
2. 列表
3. 列表

- [ ] 列表
- [ ] 列表
- [ ] 列表
```

- 列表
- 列表
- 列表

1. 列表
2. 列表
3. 列表

- [ ] 列表
- [ ] 列表
- [ ] 列表

### 多级列表

```markdown
- 一级列表
  - 二级列表
- 一级列表
```

- 一级列表
  - 二级列表
- 一级列表

### 列表中分段

```markdown
- 项目一，段落一

  项目一，段落二
```

- 项目一，段落一

  项目一，段落二

### 列表中换行

```markdown
- 项目二，第一行  
  项目二，第二行
```

- 项目二，第一行  
  项目二，第二行

## 引用

### 基本用法

```markdown
> 我亦飘零久，十年来，深恩负尽，死生师友。 ——顾贞观
```

> 我亦飘零久，十年来，深恩负尽，死生师友。 ——顾贞观

### 多级引用

```markdown
> 引用 1
>
> > 引用 2
```

> 引用 1
>
> > 引用 2

### 引用中分段

```markdown
> 引用
>
> 引用
```

> 引用
>
> 引用

### 引用中换行

```markdown
> 引用  
> 引用
```

> 引用  
> 引用

## 链接

### 文内链接

```markdown
这是一个文内链接的[例子](http://example.com/ "鼠标悬浮此处显示的标题")。

[这个](http://example.net/)链接在鼠标悬浮时没有标题。

[这个](/about/)链接是本地资源。
```

这是一个文内链接的[例子](http://example.com/ "鼠标悬浮此处显示的标题")。

[这个](http://example.net/)链接在鼠标悬浮时没有标题。

[这个](/about/)链接是本地资源。

### 引用链接

```markdown
这是一个引用链接的[例子][id]。

[id]: http://example.com/ "鼠标悬浮标题"
```

这是一个引用链接的[例子][id]。

[id]: http://example.com/ "鼠标悬浮标题"

注意，这里的 id 没有大小写区分，如果省略 id，则前面方括号的内容会被用作 id。

```markdown
我常用的网站包括[Google][1]，[Yahoo][]和[MSN][]。

[1]: http://google.com/ "Google"
[yahoo]: http://search.yahoo.com/ "Yahoo Search"
[msn]: http://search.msn.com/ "MSN Search"
```

我常用的网站包括[Google][1]，[Yahoo][]和[MSN][]。

[1]: http://google.com/ "Google"
[yahoo]: http://search.yahoo.com/ "Yahoo Search"
[msn]: http://search.msn.com/ "MSN Search"

## 图片

### 基本用法

```markdown
![mountain](./mountain.jpeg "图片下方描述")
```

![mountain](./mountain.jpeg "图片下方描述")

### base64 图片数据

```markdown
![mountain][base64str]

[base64str]: data:image/jpg;base64,xxxxxxxxxxxxxxx... "图片下方描述"
```

## 代码块

````markdown
```python
print('Hello,World')
```
````

```python
print('Hello,World')
```

## 链接

```markdown
<https://www.baidu.com/>
```

<https://www.baidu.com/>

## 转义

在不希望符号被当成 markdown 标识符时，用\转义

```markdown
\_不是斜体\_
```

\_不是斜体\_

## Karmdowm 扩展

表格

```markdown
| 左对齐 | 中间对齐 | 右对齐 |
| :----- | :------: | -----: |
| 左 1   |   中 1   |   右 1 |
| 左 2   |   中 2   |   右 3 |
```

| 左对齐 | 中间对齐 | 右对齐 |
| :----- | :------: | -----: |
| 左 1   |   中 1   |   右 1 |
| 左 2   |   中 2   |   右 3 |

## 脚注

```markdown
请参阅脚注 1. [^1]

[^1]: 脚注 1 内容。 # 出现在文末

请参阅脚注 2. [^2]

[^2]: 脚注 2 内容。 # 出现在文末
```

## HTML 扩展

### 下划线

```markdown
<u>下划内容</u>
```

<u>下划内容</u>

### 上标

```markdown
S = πr<sup>2</sup>
```

S = πr<sup>2</sup>

### 下标

```markdown
Water: H<sub>2</sub>O
```

Water: H<sub>2</sub>O

### 首行缩进

```markdown
&emsp;&emsp;开始内容
```

&emsp;&emsp;开始内容

### 内部跳转

```markdown
点此[标签](#j1)跳转。

<a name="锚点" id="j1" href="https://saltfishpr.github.io/"></a>
```

点此[标签](#j1)跳转。

<a name="锚点" id="j1" href="https://saltfishpr.github.io/"></a>

`id` 要匹配， `name` 和 `href` 都不是必须的

### 插入视频

```HTML
<div>
  <a href="//player.bilibili.com/player.html?aid=93178052&cid=159088790&page=1" target="_blank"><img src="封面图片路径" alt="没有图片时显示此文字"  width="100%" frameborder="no" framespacing="0" allowfullscreen="true" /></a>
  <script>
    var em = document.getElementById("video-iframe");
    console.log(em.clientWidth);
    em.height = em.clientWidth * 0.75
  </script>
</div>
```

<div>
  <a href="//player.bilibili.com/player.html?aid=93178052&cid=159088790&page=1" target="_blank"><img src="封面图片路径" alt="没有图片时显示此文字"  width="100%" frameborder="no" framespacing="0" allowfullscreen="true" /></a>
  <script>
    var em = document.getElementById("video-iframe");
    console.log(em.clientWidth);
    em.height = em.clientWidth * 0.75
  </script>
</div>

### 注释

```markdown
[^_^]: # (注释，不会在浏览器中显示。)
```

[^_^]: # (注释，不会在浏览器中显示。)

```markdown
<div style='display: none'>
注释内容
</div>
```

<div style='display: none'>
注释内容
</div>

```markdown
<!--
多段
注释，
不会在浏览器中显示。
-->
```

<!--
多段
注释，
不会在浏览器中显示。
-->

## short code

### person

```markdown
{{</* person url="https://saltfishpr.github.io/about/" name="硕" nick="saltfishpr" picture="https://github.com/saltfishpr.png" */>}}
```

{{< person url="https://saltfishpr.github.io/about/" name="硕" nick="saltfishpr" picture="https://github.com/saltfishpr.png" >}}

## 参考资料

1. https://www.jianshu.com/p/d7d6da4b7c60#fnref1
2. https://guides.github.com/features/mastering-markdown/
3. [emoji 目录](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md)
