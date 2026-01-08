---
title: Kubernetes
date: 2022-02-24T15:33:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: K8S 快速入门
tags: ["kubernetes"]
categories: ["Programming"]

draft: true
---

<!--more-->

## 概念

### Node（节点）

可以是一个虚拟机或者物理机器。指物理资源。

### Pods

Pod 指的是在集群上处于运行状态的一组容器。

一个 Pod 中的容器可以在不同 Node 中。

### 配置

#### ConfigMap

ConfigMap 是一种 API 对象，用来将非机密性的数据保存到键值对中。使用时，Pods 可以将其用作环境变量、命令行参数或者存储卷中的配置文件。

#### Secret

Secret 是一种包含少量敏感信息例如密码、令牌或密钥的对象。这样的信息可能会被放在 Pod 规约中或者镜像中。使用 Secret 意味着你不需要在应用程序代码中包含机密数据。

Secret 类似于 ConfigMap 但专门用于保存机密数据。

### 工作负载资源

#### Deployment

描述 Deployment 的目标状态，Deployment 控制器以受控的速率更改实际状态，使其变为期望状态。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - name: nginx
          image: nginx:1.14.2
          ports:
            - containerPort: 80
```

#### StatefulSet

如果部署的应用满足以下一个或多个部署需求，则建议使用 StatefulSet。

- 稳定的、唯一的网络标识符。
- 稳定的、持久的存储。
- 有序的、优雅的部署和扩缩。
- 有序的、自动的滚动更新。

### 网络

#### Service

Kubernetes 中 Service 是 将运行在一个或一组 Pod 上的网络应用程序公开为网络服务的方法。

应用可以通过 `<service-name>.<namespace>` 访问到服务。

例如，如果你在 Kubernetes 命名空间 `my-ns` 中有一个名为 `my-service` 的服务，则控制平面和 DNS 服务共同为 `my-service.my-ns` 创建 DNS 记录。 `my-ns` 命名空间中的 Pod 应该能够通过按名检索 `my-service` 来找到服务（`my-service.my-ns` 也有效）。

#### Ingress

Ingress 公开从集群外部到集群内服务的 HTTP 和 HTTPS 路由。流量路由由 Ingress 资源上定义的规则控制。

下面是一个将所有流量都发送到同一 Service 的简单 Ingress 示例：

![ingress.svg](https://d33wubrfki0l68.cloudfront.net/4f01eaec32889ff16ee255e97822b6d165b633f0/a54b4/zh-cn/docs/images/ingress.svg)

Ingress 不会公开任意端口或协议。将 HTTP 和 HTTPS 以外的服务公开到 Internet 时，通常使用 `Service.Type=NodePort` 或 `Service.Type=LoadBalancer` 类型的 Service。

#### Ingress 控制器

为了让 Ingress 资源工作，集群必须有一个正在运行的 Ingress 控制器。

[ingress-nginx](https://github.com/kubernetes/ingress-nginx/blob/main/README.md#readme)

## minikube

minikube 是本地的 Kubernetes，专注于让 Kubernetes 易于学习和开发。

创建集群，指定 CPU 数量

```shell
minikube start --cpus=4
```

启动 dashboard

```shell
minikube dashboard
```

## kubectl

`kubectl` 是使用 Kubernetes API 与 Kubernetes 集群的控制面进行通信的命令行工具。

查看一个 deployment 所有 Pod 的日志

```shell
kubectl logs -f -l app=echo-app --all-containers
```

## helm

Helm 是 Kubernetes 的包管理器

添加 bitnami helm chart 库

```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
```

安装 dashboard 插件

```shell
helm plugin install https://github.com/komodorio/helm-dashboard.git
```

启动 dashboard

```shell
helm dashboard
```
