---
title: Postgres 数据库使用
date: 2021-11-24T21:42:55+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
tags: ["database", "postgres"]
categories: ["Programming"]

draft: true
---

<!--more-->

创建用户

```sql
create user guest with password '123456';
```

创建数据库并指定 owner

```sql
create database chat owner guest;
```

开启 UUID 生成插件

```sql
create extension if not exists "uuid-ossp";
```
