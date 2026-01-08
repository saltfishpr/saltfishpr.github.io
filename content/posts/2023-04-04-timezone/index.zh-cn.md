---
title: Time Zone && Offset
date: 2023-04-04T15:24:45Z
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: 时区与偏移量的关系
tags: []
categories: ["Programming"]
---

<!--more-->

## Time Zone 和 Offset

时区不能仅用与 UTC 的偏移量来表示。由于夏令时（又称“夏令时”）的规定，许多时区有多个偏移量。偏移量更改的日期也是时区规则的一部分，任何历史偏移量更改也是如此。许多软件程序、库和 web 服务忽略了这一重要细节，并错误地将标准或当前偏移量称为“区域”。这可能会导致数据的混乱和滥用。请尽可能使用正确的术语。

偏移量只是一个数字，表示特定日期/时间值领先或落后 UTC 的程度

- 大多数偏移量以整小时表示。
- 但也有许多是 30 分钟的偏移。
- 还有一些是 45 分钟的偏移。

时区包含更多内容

- 可用于标识区域的名称或 ID。
- 与 UTC 的一个或多个偏移。
- 分区在偏移之间转换的特定日期和时间。
- 有时，可以呈现给用户的单独的特定语言的显示名称。

{{< admonition note >}}
在给定时区和 Unix 时间戳的情况下，可以确定正确的偏移量。但是，如果只给出一个偏移量，无法确定正确的时区。
{{< /admonition >}}

## eztz

[eztz](https://github.com/saltfishpr/go-learning/tree/examples/timezone) 是一个时间转换工具，将给定的时间字符串和时区转换成时间戳。时区数据从 timezonedb 获取，后备选项为 golang 内嵌的时区数据。用户也可以添加自定义的时区数据

- [ ] 时区数据来源优先级

## JS

获取用户时区

```js
const locale = new Intl.Locale(window.navigator.language);
```

获取可用的时区

```js
const timeZones = Intl.supportedValuesOf("timeZone");
```
