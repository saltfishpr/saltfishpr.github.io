---
title: Python 字符串和编码
date: 2020-03-06T00:00:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: Python 中的字符串和编码问题
tags: ["python"]
categories: ["Programming"]
---

<!--more-->

## 编码

编码是数据从一种格式变为另一种格式的过程。通过编码，我们可以把数据以不同的格式保存和转移。

`unicode` 是一个字符集，为每一个字符分配一个十六进制数字 `0x0000` ~ `0xFFFF`。

`utf-8` 是一种编码规则，它将一个 `unicode` 字符以 8 位为一个单位编码。

## Python3 字符串类型

unicode --encode("utf-8")-> UTF-8 编码的二进制数据

UTF-8 编码的二进制数据 --decode("utf-8")-> unicode

示例 1：

```python
s1 = "中文"
s2 = s1.encode("utf-8")
print(type(s1), s1)
print(type(s2), s2)
```

output 1:

```
<class 'str'> 中文
<class 'bytes'> b'\xe4\xb8\xad\xe6\x96\x87'
```

即：`中`的 `utf-8` 编码为 `0xe4 0xb8 0xad`

文件中存储的都是 byte 这样一个一个二进制数，所以在将 str 存入文件或从文件读取内容时需要指明编码类型。

## 经验总结

> 写 python 程序的时候，把编码和解码操作放在界面的最外围来做，程序的核心部分使用 Unicode 字符类型 (Python3 中的 str)。
>
> <p align="right">——Effective+Python</p>

1. 在写文件时注明编码类型

   ```python
   with open(file_path, 'w',encode='utf-8') as f:
       f.write(data)
   ```

2. 在使用 `requests` 库获取响应信息时，指明编码类型

   ```python
   def send_req(url):
       headers = {}
       req = requests.get(url, headers)
       req.encoding = "utf-8"
       if req.ok:
           print(req.text)
           return req.text
       else:
           return Exception("访问失败")
   ```

3. base64 编码

   Base64 是网络上最常见的用于传输 8Bit 字节码的编码方式之一，可用于在 HTTP 环境下传递较长的标识信息。采用 Base64 编码具有不可读性，需要解码后才能阅读。

   将文件转换成 base64 编码形式进行传输。

   ```python
   import base64
   with open(file_path, 'rb') as f:
      res = base64.b64encode(f.read())
   ```

4. python3 中 str 和 bytes 互换函数

   ```python
   def to_str(bytes_or_str):
       if isinstance(bytes_or_str, bytes):
               value = bytes_or_str.decode('utf-8')
           else:
               value = bytes_or_str
           return value

   def to_bytes(bytes_or_str):
       if isinstance(bytes_or_str, str):
               value = bytes_or_str.encode('utf-8')
           else:
               value = bytes_or_str
           return value
   ```

参考资料：

1. [base64 编码](https://baike.baidu.com/item/base64/8545775?fr=aladdin "百度百科：base64编码")
2. Effective+Python
