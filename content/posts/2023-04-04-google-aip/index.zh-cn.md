---
title: Google AIP
date: "2023-04-04T16:03:42Z"
author: Salt Fish
authorLink: https://github.com/saltfishpr
tags: ["api design"]
categories: ["Programming"]
---

[AIP](https://google.aip.dev/general) (API Improvement Proposals) 是总结 Google API 设计决策的设计文档。

这里提供了一些常用文档的翻译。

<!--more-->

## **AIP-1** AIP 目的和指南 {#AIP-1}

{{< admonition info >}}
翻译时间：2023-04-05
{{< /admonition >}}

随着谷歌 API 语料库的发展，以及 API 治理团队的壮大，以满足支持它们的需求，越来越有必要为 API 生产者、审查者和其他相关方提供文档语料库以供参考。API 风格指南和介绍性的 One Platform 文档有意简洁而高级。AIP 集合提供了一种为 API 设计指南提供一致文档的方法。

## **AIP-131** 标准方法：获取（Standard methods: Get） {#AIP-131}

{{< admonition info >}}
翻译时间：2023-04-05
{{< /admonition >}}

在 REST API 中，为了检索资源，通常会向资源的 URI（例如 `/v1/publisher/{publisher}/books/{book}`）发出 `GET` 请求。

面向资源的设计（[AIP-121](#AIP-121)）通过 `Get` 方法来尊重这种模式。这些 RPC 接受表示该资源的 URI 并返回该资源。

### 指导

API <font color="#c5221f">必须</font>为资源提供一个 get 方法。get 方法的目的是从单个资源返回数据。

Get 方法是使用以下模式指定的：

```protobuf
rpc GetBook(GetBookRequest) returns (Book) {
  option (google.api.http) = {
    get: "/v1/{name=publishers/*/books/*}"
  };
  option (google.api.method_signature) = "name";
}
```

- RPC 的名称<font color="#c5221f">必须</font>以 `Get` 一词开头。RPC 名称的其余部分<font color="#f57c00">应该</font>是资源消息名称的单数形式。
- 请求消息<font color="#c5221f">必须</font>与 RPC 名称匹配，并带有 `Request` 后缀。
- 响应消息<font color="#c5221f">必须</font>是资源本身。（没有 `GetBookResponse`。）
  - 响应通常应包括完全填充的资源，除非有理由返回部分响应（见 [AIP-157](#AIP-157)）。
- HTTP 谓词<font color="#c5221f">必须</font>是 `GET`。
- URI <font color="#f57c00">应该</font>包含与资源名称相对应的单个变量字段。
  - 此字段应称为 `name`。
  - URI <font color="#f57c00">应该</font>有一个与此字段相对应的变量。
  - `name` 字段<font color="#f57c00">应该</font>是 URI 路径中唯一的变量。所有剩余的参数都<font color="#f57c00">应该</font>映射到 URI 查询参数。
- `google.api.http` 注解中<font color="#c5221f">不能</font>有 `body` 键。
- <font color="#f57c00">应该</font>只有一个值为 `"name"` 的 `google.api.method_signature` 注解。

### 请求消息

Get 方法实现了一个通用的请求消息模式：

```protobuf
message GetBookRequest {
  // The name of the book to retrieve.
  // Format: publishers/{publisher}/books/{book}
  string name = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      type: "library.googleapis.com/Book"
    }];
}
```

- <font color="#c5221f">必须</font>包含资源名称字段。它<font color="#f57c00">应该</font>被称为 name。
  - <font color="#f57c00">应该</font>根据需要对字段进行[注释](#AIP-203)。
  - 字段<font color="#c5221f">必须</font>标识其引用的[资源类型](#AIP-123)。
- `name` 字段的注释<font color="#f57c00">应该</font>记录资源模式。
- 请求消息<font color="#c5221f">不得</font>包含任何其他必需字段，也<font color="#f57c00">不应</font>包含除其他 AIP 中描述的字段之外的其他可选字段。

{{< admonition note >}}
请求对象中的 `name` 字段对应于 RPC 上 `google.api.http` 注解中的 `name` 变量。因为在使用 REST/JSON 接口时，根据 URL 中的值填充 request 中的 `name` 字段。
{{< /admonition >}}

### 错误

请参阅[errors](#AIP-193)，特别是何时使用 `PERMISSION_DENIED` 和 `NOT_FOUND` 错误。

### Changelog

- 2023-03-17: Align with AIP-122 and make Get a must.
- 2022-11-04: Aggregated error guidance to AIP-193.
- 2022-06-02: Changed suffix descriptions to eliminate superfluous "-".
- 2020-06-08: Added guidance on returning the full resource.
- 2019-10-18: Added guidance on annotations.
- 2019-08-12: Added guidance for error cases.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to present a better example of resource ownership.
- 2019-05-29: Added an explicit prohibition on arbitrary fields in standard methods.

## **AIP-132** 标准方法：列表（Standard methods: List） {#AIP-132}

{{< admonition info >}}
翻译时间：2023-04-05
{{< /admonition >}}

在许多 API 中，通常会向集合的 URI（例如 `/v1/publisher/1/books`）发出 `GET` 请求，以便检索资源列表，每个资源都位于该集合中。

面向资源的设计（[AIP-121](#AIP-121)）通过 `List` 方法来尊重这种模式。这些 RPC 接受父集合（可能还有一些其他参数），并返回与该输入匹配的响应列表。

### 指导

API <font color="#c5221f">必须</font>为资源提供 `List` 方法，除非资源是[单例](#AIP-156)。`List` 方法的目的是从有限集合返回数据（通常是单个集合，除非操作支持[跨集合读取](#AIP-159)）。

List 方法是使用以下模式指定的：

```protobuf
rpc ListBooks(ListBooksRequest) returns (ListBooksResponse) {
  option (google.api.http) = {
    get: "/v1/{parent=publishers/*}/books"
  };
  option (google.api.method_signature) = "parent";
}
```

- RPC 的名称<font color="#c5221f">必须</font>以单词 `List` 开头。RPC 名称的其余部分<font color="#f57c00">应该</font>是所列资源的复数形式。
- 请求和响应消息<font color="#c5221f">必须</font>与 RPC 名称匹配，并带有 `Request` 和 `Response` 后缀。
- HTTP 谓词<font color="#c5221f">必须</font>是 `GET`。
- 列出其资源的集合<font color="#f57c00">应该</font>映射到 URI 路径。
  - 集合的父资源<font color="#f57c00">应</font>称为 `parent`，并且<font color="#f57c00">应</font>是 URI 路径中唯一的变量。所有剩余的参数都<font color="#f57c00">应该</font>映射到 URI 查询参数。
  - 集合标识符（上例中的 books）<font color="#c5221f">必须</font>是一个文本字符串。
- `google.api.http` 注解中的 `body` 键<font color="#c5221f">必须</font>省略。
- 如果列出的资源不是顶级资源，那么<font color="#f57c00">应该</font>只有一个值为 `"parent"` 的 `google.api.method_signature` 注解。如果列出的资源是顶级资源，则<font color="#f57c00">应该</font>没有 `google.api.method_signature` 注解，或者只有一个值为 `""` 的 `google.api.method_signature` 注解。

### 请求消息

List 方法实现了一种常见的请求消息模式：

```protobuf
message ListBooksRequest {
  // The parent, which owns this collection of books.
  // Format: publishers/{publisher}
  string parent = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      child_type: "library.googleapis.com/Book"
    }];

  // The maximum number of books to return. The service may return fewer than
  // this value.
  // If unspecified, at most 50 books will be returned.
  // The maximum value is 1000; values above 1000 will be coerced to 1000.
  int32 page_size = 2;

  // A page token, received from a previous `ListBooks` call.
  // Provide this to retrieve the subsequent page.
  //
  // When paginating, all other parameters provided to `ListBooks` must match
  // the call that provided the page token.
  string page_token = 3;
}
```

- 除非列出的资源是顶级资源，否则<font color="#c5221f">必须</font>包括 `parent` 字段。它应该被称为 `parent`。
  - <font color="#f57c00">应</font>根据需要对字段进行[注释](#AIP-203)。
  - 字段<font color="#c5221f">必须</font>标识所列资源的[资源类型](#AIP-123)。
- <font color="#c5221f">必须</font>在所有列表请求消息上指定支持分页的 `page_size` 和 `page_token` 字段。有关更多信息，请参阅 [AIP-158](#AIP-158)。
  - `page_size` 字段上方的注释<font color="#f57c00">应</font>记录允许的最大值，以及省略（或设置为 `0`）字段时的默认值。如果首选 (_If preferred_)，API <font color="#188038">可以</font>声明服务器将使用合理的默认值。此默认值<font color="#188038">可</font>会随着时间的推移而更改。
  - 如果用户提供的值大于允许的最大值，则 API <font color="#f57c00">应</font>将该值强制为允许的最大。
  - 如果用户提供了负值或其他无效值，则 API <font color="#c5221f">必须</font>返回 `INVALID_ARGUMENT` 错误。
- `page_token` 字段<font color="#c5221f">必须</font>包含在所有列表请求消息中。
- 请求消息<font color="#188038">可以</font>包括与列表方法相关的常见设计模式的字段，例如 `string filter` 和 `string order_by`。
- 请求消息<font color="#c5221f">不得</font>包含任何其他必需字段，也<font color="#f57c00">不应</font>包含除本 AIP 或其他 AIP 中描述的字段之外的其他可选字段。

{{< admonition note >}}
对于任何有权对集合成功发出 List 请求的用户，List 方法都<font color="#f57c00">应该</font>返回相同的结果。Search 方法在这方面比较宽松。
{{< /admonition >}}

### 响应消息

List 方法实现了一个通用的响应消息模式：

```protobuf
message ListBooksResponse {
  // The books from the specified publisher.
  repeated Book books = 1;

  // A token, which can be sent as `page_token` to retrieve the next page.
  // If this field is omitted, there are no subsequent pages.
  string next_page_token = 2;
}
```

- 响应消息<font color="#c5221f">必须</font>包括一个与返回的资源相对应的重复字段，并且<font color="#f57c00">不应</font>包括任何其他重复字段，除非在另一个 AIP（例如，[AIP-217](#AIP-217)）中描述。
  - 响应通常<font color="#f57c00">应</font>包括完全填充的资源，除非有理由返回部分响应（见 [AIP-157](#AIP-157)）。
- 支持分页的 `next_page_token` 字段<font color="#c5221f">必须</font>包含在所有列表响应消息中。如果有后续页面，则<font color="#c5221f">必须</font>设置它，如果响应表示最终页面，则<font color="#c5221f">不得</font>设置它。有关更多信息，请参阅 [AIP-158](#AIP-158)。
- 该消息<font color="#188038">可以</font>包括具有集合中的项数的 `int32 total_size`（或 `int64 total_size`）字段。
  - 该值<font color="#188038">可以</font>是一个估计值（如果是的话，字段<font color="#f57c00">应该</font>清楚地记录在文档中）。
  - 如果使用筛选，则 `total_size` 字段<font color="#f57c00">应</font>反映应用筛选后集合的大小。

### 排序

`List` 方法<font color="#188038">可以</font>允许客户端指定排序顺序；如果他们这样做了，那么请求消息<font color="#f57c00">应该</font>包含一个 `string order_by` 字段。

- 值<font color="#f57c00">应该</font>是以逗号分隔的字段列表。例如：`"foo,bar"`。
- 默认的排序顺序是升序。为了指定字段的降序，用户会附加一个 `" desc"` 后缀；例如：`"foo desc, bar"`。
- 语法中多余的空格字符是无关紧要的。`"foo, bar desc"`、`" foo , bar desc "`和 `"foo,bar desc"` 都是等价的。
- 子字段是用指定的。字符，例如 `foo.bar` 或 `address.street`。

{{< admonition note >}}
只有在确定需要的情况下才提供排序。以后随时可以添加排序，但删除它是一个破坏性的改变。
{{< /admonition >}}

### 过滤

`List` 方法<font color="#188038">可以</font>允许客户端指定筛选器；如果他们这样做了，那么请求消息应该包含一个 `string filter` 字段。过滤在 [AIP-160](#AIP-160) 中有更详细的描述。

{{< admonition note >}}
只有在确定需要的情况下才提供筛选。以后随时是可以添加筛选，但删除它是一个破坏性的改变。
{{< /admonition >}}

### 软删除的资源

一些 API 需要“[软删除](#AIP-135)”资源，将其标记为已删除或待删除（并可选择稍后清除）。

默认情况下，执行此操作的 API <font color="#f57c00">不应</font>在列表请求中包括已删除的资源。具有软删除资源的 API <font color="#f57c00">应</font>在列表请求中包括一个 `bool show_deleted` 字段，如果设置该字段，将导致软删除的资源被包括在内。

### 错误

请参阅[errors](#AIP-193)，特别是何时使用 `PERMISSION_DENIED` 和 `NOT_FOUND` 错误。

### 延伸阅读

- 有关分页的详细信息，请参阅 [AIP-158](#AIP-158)。
- 有关跨多个父集合的列表，请参见 [AIP-159](#AIP-159)。

### Changelog

- 2023-03-22: Fix guidance wording to mention AIP-159.
- 2023-03-17: Align with AIP-122 and make Get a must.
- 2022-11-04: Aggregated error guidance to AIP-193.
- 2022-06-02: Changed suffix descriptions to eliminate superfluous "-".
- 2020-09-02: Add link to the filtering AIP.
- 2020-08-14: Added error guidance for permission denied cases.
- 2020-06-08: Added guidance on returning the full resource.
- 2020-05-19: Removed requirement to document ordering behavior.
- 2020-04-15: Added guidance on List permissions.
- 2019-10-18: Added guidance on annotations.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to present a better example of resource ownership.
- 2019-07-30: Added guidance about documenting the ordering behavior.
- 2019-05-29: Added an explicit prohibition on arbitrary fields in standard methods.

## **AIP-133** 标准方法：创建（Standard methods: Create） {#AIP-133}

{{< admonition info >}}
翻译时间：2023-04-05
{{< /admonition >}}

在 REST API 中，通常会向集合的 URI（例如 `/v1/publisher/{publisher}/books`）发出 `POST` 请求，以便在该集合中创建新的资源。

面向资源的设计（[AIP-121](#AIP-121)）通过 `Create` 方法来尊重这种模式。这些 RPC 接受父集合和要创建的资源（可能还有一些其他参数），并返回创建的资源。

### 指导

API 通常<font color="#f57c00">应该</font>为资源提供一个*创建*方法，除非这样做对用户来说没有价值。*创建*方法的目的是在已经存在的集合中创建一个新的资源。

Create 方法是使用以下模式指定的：

```protobuf
rpc CreateBook(CreateBookRequest) returns (Book) {
  option (google.api.http) = {
    post: "/v1/{parent=publishers/*}/books"
    body: "book"
  };
  option (google.api.method_signature) = "parent,book";
}
```

- RPC 的名称<font color="#c5221f">必须</font>以 `Create` 一词开头。RPC 名称的其余部分<font color="#f57c00">应该</font>是正在创建的资源的单数形式。
- 请求消息<font color="#c5221f">必须</font>与 RPC 名称匹配，并带有 `Request` 后缀。
- 响应消息<font color="#c5221f">必须</font>是资源本身。没有 `CreateBookResponse`。
  - 响应<font color="#f57c00">应</font>包括完全填充的资源，并且<font color="#c5221f">必须</font>包括所提供的任何字段，除非它们仅是输入的（见 [AIP-203](#AIP-203)）。
  - 如果创建 RPC 是长时间运行的，则响应消息<font color="#c5221f">必须</font>是解析为资源本身的 `google.longrunning.Operation`。
- HTTP 谓词<font color="#c5221f">必须</font>是 `POST`。
- 添加资源的集合<font color="#f57c00">应该</font>映射到 URI 路径。
  - 集合的父资源<font color="#f57c00">应</font>称为 `parent`，并且<font color="#f57c00">应</font>是 URI 路径中唯一的变量。
  - 集合标识符（上例中的 `books`）<font color="#c5221f">必须</font>是一个文本字符串。
- `google.api.http` 注解中<font color="#c5221f">必须</font>有一个 `body` 键，并且它<font color="#c5221f">必须</font>映射到请求消息中的资源字段。
  - 所有剩余的字段都<font color="#f57c00">应该</font>映射到 URI 查询参数。
- <font color="#f57c00">应该</font>只有一个 `google.api.method_signature` 注解，如果要创建的资源不是顶级资源，则其值为 `"parent，{resource}"`，如果正在创建的资源是顶级资源（除非该方法支持[用户指定的 ID](#AIP-133)），则该注解的值为 `"{resource}"`。

### 请求消息

Create 方法实现一个通用的请求消息模式：

```protobuf
message CreateBookRequest {
  // The parent resource where this book will be created.
  // Format: publishers/{publisher}
  string parent = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      child_type: "library.googleapis.com/Book"
    }];

  // The book to create.
  Book book = 2 [(google.api.field_behavior) = REQUIRED];
}
```

- 除非正在创建的资源是顶级资源，否则<font color="#c5221f">必须</font>包括 `parent` 字段。它<font color="#f57c00">应该</font>被称为 `parent`。
  - <font color="#f57c00">应</font>根据需要对字段进行[注释](#AIP-203)。
  - 字段<font color="#c5221f">必须</font>标识正在创建的资源的[资源类型](#AIP-123)。
- 资源字段<font color="#c5221f">必须</font>包含在内，并且<font color="#c5221f">必须</font>映射到 POST body。
- 请求消息<font color="#c5221f">不得</font>包含任何其他必需字段，也<font color="#f57c00">不应</font>包含除本 AIP 或其他 AIP 中描述的字段之外的其他可选字段。

### 长时间运行的创建（Long-running create）

有些资源创建资源所花费的时间比常规 API 请求所需的时间更长。在这种情况下，API <font color="#f57c00">应该</font>使用长时间运行的操作（[AIP-151](#AIP-151)）：

```protobuf
rpc CreateBook(CreateBookRequest) returns (google.longrunning.Operation) {
  option (google.api.http) = {
    post: "/v1/{parent=publishers/*}/books"
  };
  option (google.longrunning.operation_info) = {
    response_type: "Book"
    metadata_type: "OperationMetadata"
  };
}
```

- 响应类型<font color="#c5221f">必须</font>设置为资源（如果 RPC 不是长时间运行的，则返回类型是什么）。
- <font color="#c5221f">必须</font>同时指定 `response_type` 和 `metadata_type` 字段。

{{< admonition warning "重要" >}}
声明性友好资源（[AIP-128](#AIP-128)）<font color="#f57c00">应该</font>使用长时间运行的操作。如果请求实际上是即时的，则服务<font color="#188038">可以</font>返回已经设置为完成的 LRO。
{{< /admonition >}}

### 用户指定的 ID

有时，API 需要允许客户端在创建时指定资源的 ID 组件（资源名称的最后一段）。如果允许用户选择其资源名称的这一部分，这是很常见的。

例如：

```plaintext
// Using user-specified IDs.
publishers/lacroix/books/les-miserables

// Using system-generated IDs.
publishers/012345678-abcd-cdef/books/12341234-5678-abcd
```

创建 RPC <font color="#188038">可以</font>通过在请求消息上提供字符串 `{resource}_id` 字段来支持此行为：

```protobuf
message CreateBookRequest {
  // The parent resource where this book will be created.
  // Format: publishers/{publisher}
  string parent = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      child_type: "library.googleapis.com/Book"
    }];

  // The book to create.
  Book book = 2 [(google.api.field_behavior) = REQUIRED];

  // The ID to use for the book, which will become the final component of
  // the book's resource name.
  //
  // This value should be 4-63 characters, and valid characters
  // are /[a-z][0-9]-/.
  string book_id = 3;
}
```

- `{resource}_id` 字段<font color="#c5221f">必须</font>存在于请求消息中，而不是资源本身。
  - 该字段<font color="#188038">可以</font>是必需的，也可以是可选的。如果是必需的，它<font color="#f57c00">应该</font>包括相应的注释。
- <font color="#c5221f">必须</font>忽略资源上的 `name` 字段。
- RPC 上<font color="#f57c00">应该</font>只有一个 `google.api.method_signature` 注解，如果正在创建的资源不是顶级资源，则其值为 `"parent,{resource},{resource}_id"`；如果要创建的资源是顶级资源，其值应为 `"{resource},{resource}_id"`。
- 文档<font color="#f57c00">应</font>解释什么是可接受的格式，并且该格式<font color="#f57c00">应</font>遵循 [AIP-122](#AIP-122) 中的资源名称格式指南。
- 如果用户试图创建一个 ID 会导致资源名称重复的资源，则该服务<font color="#c5221f">必须</font>返回 `ALREADY_EXISTS` 错误。
  - 但是，如果进行调用的用户没有查看重复资源的权限，则服务<font color="#c5221f">必须</font>改为使用 `PERMISSION_DENIED`。

{{< admonition note >}}
对于 REST API，用户指定的 ID 字段 `{resource}_id` 作为请求 URI 上的查询参数提供。
{{< /admonition >}}

{{< admonition warning "重要">}}
声明性友好资源（[AIP-128](#AIP-128)）<font color="#c5221f">必须</font>支持用户指定的 ID。
{{< /admonition >}}

### 错误

请参阅[errors](#AIP-193)，特别是何时使用 `PERMISSION_DENIED` 和 `NOT_FOUND` 错误。

### 延伸阅读

- 要确保 `Create` 方法中的幂等性，请参见 [AIP-155](#AIP-155)。
- 有关涉及 Unicode 的命名资源，请参见 [AIP-210](#AIP-210)。

### Changelog

- 2022-11-04: Referencing aggregated error guidance in AIP-193, similar to other CRUDL AIPs.
- 2022-06-02: Changed suffix descriptions to eliminate superfluous "-".
- 2020-10-06: Added declarative-friendly guidance.
- 2020-08-14: Updated error guidance to use permission denied over forbidden.
- 2020-06-08: Added guidance on returning the full resource.
- 2019-11-22: Added clarification on what error to use if a duplicate name is sent.
- 2019-10-18: Added guidance on annotations.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to present a better example of resource ownership.
- 2019-06-10: Added guidance for long-running create.
- 2019-05-29: Added an explicit prohibition on arbitrary fields in standard methods.

## **AIP-134** 标准方法：更新（Standard methods: Update） {#AIP-134}

{{< admonition info >}}
翻译时间：2023-04-05
{{< /admonition >}}

在 REST API 中，通常对资源的 URI（例如 `/v1/publisher/{publisher}/books/{book}`）发出 `PATCH` 或 `PUT` 请求，以更新该资源。

面向资源的设计（[AIP-121](#AIP-121)）通过 `Update` 方法（反映 REST `PATCH` 行为）来尊重这种模式。这些 RPC 接受表示该资源的 URI 并返回该资源。

### 指导

API 通常<font color="#f57c00">应该</font>为资源提供更新方法，除非这样做对用户没有价值。更新方法的目的是在不造成副作用的情况下对资源进行更改。

Update 方法是使用以下模式指定的：

```protobuf
rpc UpdateBook(UpdateBookRequest) returns (Book) {
  option (google.api.http) = {
    patch: "/v1/{book.name=publishers/*/books/*}"
    body: "book"
  };
  option (google.api.method_signature) = "book,update_mask";
}
```

- RPC 的名称<font color="#c5221f">必须</font>以单词 `Update` 开头。RPC 名称的其余部分<font color="#f57c00">应该</font>是资源消息名称的单数形式。
- 请求消息<font color="#c5221f">必须</font>与 RPC 名称匹配，并带有 `Request` 后缀。
- 响应消息<font color="#c5221f">必须</font>是资源本身。（没有 `UpdateBookResponse`。）
  - 响应<font color="#f57c00">应</font>包括完全填充的资源，并且<font color="#c5221f">必须</font>包括已发送并包含在更新掩码中的任何字段，除非它们只是输入的（请参阅 [AIP-203](#AIP-203)）。
  - 如果更新 RPC 是[长时间运行的](#AIP-134)，则响应消息<font color="#c5221f">必须</font>是解析为资源本身的 `google.longrunning.Operation`。
- 该方法<font color="#f57c00">应该</font>支持部分资源更新，HTTP 谓词应该是 `PATCH`。
  - 如果该方法只支持完全资源替换，那么 HTTP 谓词<font color="#188038">可以</font>是 `PUT`。然而，强烈反对这样做，因为向资源中添加字段会成为向后不兼容的更改。
- 资源的 `name` 字段<font color="#f57c00">应该</font>映射到 URI 路径。
  - `{resource}.name` 字段<font color="#f57c00">应该</font>是 URI 路径中的唯一变量。
- `google.api.http` 注解中<font color="#c5221f">必须</font>有一个 `body` 键，并且它<font color="#c5221f">必须</font>映射到请求消息中的 resource 字段。
  - 所有剩余的字段都<font color="#f57c00">应该</font>映射到 URI 查询参数。
- <font color="#f57c00">应该</font>只有一个 `google.api.method_signature` 注解，其值为 `"{resource},update_mask"`。

{{< admonition note >}}
与其他四种标准方法不同，这里的 URI 路径引用了示例中的嵌套字段（`book.name`）。如果资源字段有单词分隔符，则使用 `snake_case`。
{{< /admonition >}}

### 请求消息

Update 方法实现了一个通用的请求消息模式：

```protobuf
message UpdateBookRequest {
  // The book to update.
  //
  // The book's `name` field is used to identify the book to update.
  // Format: publishers/{publisher}/books/{book}
  Book book = 1 [(google.api.field_behavior) = REQUIRED];

  // The list of fields to update.
  google.protobuf.FieldMask update_mask = 2;
}
```

- 请求消息<font color="#c5221f">必须</font>包含资源的字段。
  - 字段<font color="#c5221f">必须</font>映射到 `PATCH` 主体。
  - <font color="#f57c00">应</font>根据需要对字段进行[注释](#AIP-203)。
  - 资源消息中<font color="#c5221f">必须</font>包含 `name` 字段。它<font color="#f57c00">应该</font>被称为 `name`。
  - 该字段<font color="#c5221f">必须</font>标识要更新的资源的[资源类型](#AIP-123)。
- 如果支持部分资源更新，则<font color="#c5221f">必须</font>包含字段掩码。它的类型<font color="#c5221f">必须</font>是 `google.protobuf.FieldMask`，并且应该称为 `update_mask`。
  - 字段掩码中使用的字段对应于正在更新的资源（而不是请求消息）。
  - 该字段<font color="#188038">可以</font>是必需的，也可以是可选的。如果是必须的，则<font color="#c5221f">必须</font>包含相应的注释。如果可选，则服务<font color="#c5221f">必须</font>将省略的字段掩码视为隐含字段掩码，该隐含字段掩码等效于填充的所有字段（具有非空值）。
  - 更新掩码<font color="#c5221f">必须</font>支持一个特殊值 `*`，这意味着完全替换（相当于 `PUT`）。
- 请求消息<font color="#c5221f">不得</font>包含任何其他必需字段，也<font color="#f57c00">不应</font>包含除本 AIP 或其他 AIP 中描述的字段之外的其他可选字段。

### 副作用

通常，更新方法旨在更新资源中的数据。更新方法<font color="#f57c00">不应</font>引发其他副作用。相反，副作用<font color="#f57c00">应该</font>由自定义方法触发。

特别是，这要求[状态](#AIP-216)字段在更新方法中<font color="#c5221f">不能</font>直接写入。

### PATCH 和 PUT

{{< admonition tip "TL;DR" >}}
Google API 通常只使用 `PATCH` HTTP 谓词，不支持 `PUT` 请求。
{{< /admonition >}}

我们在 `PATCH` 上实现了标准化，因为谷歌通过向后兼容的改进来更新稳定的 API。通常有必要向现有资源中添加一个新字段，但在使用 `PUT` 时，这将成为一个破坏性的更改。

为了说明这一点，考虑对 `Book` 资源的 `PUT` 请求：

```plaintext
PUT /v1/publishers/123/books/456

{"title": "Mary Poppins", "author": "P.L. Travers"}
```

接下来，考虑资源稍后会添加一个新字段（此处我们添加 `rating`）：

```protobuf
message Book {
  string title = 1;
  string author = 2;

  // Subsequently added to v1 in place...
  int32 rating = 3;
}
```

如果对一本书设置了评级，并且执行了现有的 `PUT` 请求，那么它将清除该书的评级。从本质上讲，`PUT` 请求无意中擦除了数据，因为以前的版本并不知道这一点。

### 长时间运行的更新（Long-running update）

某些资源更新资源所需的时间比常规 API 请求所需的合理时间要长。在这种情况下，API <font color="#f57c00">应该</font>使用长时间运行的操作（[AIP-151](#AIP-151)）：

```protobuf
rpc UpdateBook(UpdateBookRequest) returns (google.longrunning.Operation) {
  option (google.api.http) = {
    patch: "/v1/{book.name=publishers/*/books/*}"
  };
  option (google.longrunning.operation_info) = {
    response_type: "Book"
    metadata_type: "OperationMetadata"
  };
}
```

- 响应类型<font color="#c5221f">必须</font>设置为资源（如果 RPC 不是长时间运行的，则返回类型是什么）。
- <font color="#c5221f">必须</font>同时指定 `response_type` 和 `metadata_type` 字段。

{{< admonition note >}}
声明性友好资源（[AIP-128](#AIP-128)）<font color="#f57c00">应该</font>使用长时间运行的更新
{{< /admonition >}}

### 创建或更新

如果服务使用客户端分配的资源名称，`Update` 方法<font color="#188038">可以</font>暴露一个 `bool allow_missing` 字段，这将使该方法在用户尝试更新不存在的资源时成功（并将在此过程中创建资源）：

```protobuf
message UpdateBookRequest {
  // The book to update.
  //
  // The book's `name` field is used to identify the book to be updated.
  // Format: publishers/{publisher}/books/{book}
  Book book = 1 [(google.api.field_behavior) = REQUIRED];

  // The list of fields to be updated.
  google.protobuf.FieldMask update_mask = 2;

  // If set to true, and the book is not found, a new book will be created.
  // In this situation, `update_mask` is ignored.
  bool allow_missing = 3;
}
```

更具体地说，`allow_missing` 标志触发以下行为：

- 如果方法调用所在的资源不存在，则会创建该资源。应用所有字段，而不考虑提供的任何字段掩码。
  - 但是，如果缺少任何必需的字段或字段的值无效，则会返回 `INVALID_ARGUMENT` 错误。
- 如果方法调用位于已存在的资源上，并且所有字段都匹配，则返回的现有资源不变。
- 如果方法调用位于已存在的资源上，则只更新字段掩码中声明的字段。

即使 `allow_missing` 设置为 `true`，用户也<font color="#c5221f">必须</font>具有调用 `Update` 的更新权限。对于希望阻止用户使用更新方法创建资源的客户，<font color="#f57c00">应</font>使用 IAM（Identity and Access Management）条件。

{{< admonition note >}}
声明性友好资源（[AIP-128](#AIP-128)）<font color="#c5221f">必须</font>公开 `bool allow_missing` 字段。
{{< /admonition >}}

### Etags

API 有时可能需要允许用户发送更新请求，这些更新请求保证是根据最新数据进行的（这方面的一个常见用例是检测和避免竞争条件）。需要启用此功能的资源通过包含一个 `string etag` 字段来实现，该字段包含一个不透明的、由服务器计算的值，表示资源的内容。

在这种情况下，资源<font color="#f57c00">应该</font>包含一个 `string etag` 字段：

```protobuf
message Book {
  // The resource name of the book.
  // Format: publishers/{publisher}/books/{book}
  string name = 1;

  // The title of the book.
  // Example: "Mary Poppins"
  string title = 2;

  // The author of the book.
  // Example: "P.L. Travers"
  string author = 3;

  // The etag for this book.
  // If this is provided on update, it must match the server's etag.
  string etag = 4;
}
```

etag 字段<font color="#188038">可以</font>是必需的，也可以是可选的。如果设置了它，则<font color="#c5221f">只有</font>当且仅当提供的 etag 与服务器计算的值匹配时，请求才能成功，否则<font color="#c5221f">必须</font>以 `ABORTED` 错误失败。请求中的 `update_mask` 字段不会影响 `etag` 字段的行为，因为它不是*正在*更新的字段。

### 昂贵的字段

API 有时会遇到资源上的某些字段昂贵或无法可靠返回的情况。

这种情况可能发生在几种情况下：

- 资源可能具有一些计算成本非常高的字段，并且这些字段通常对客户的更新请求没有用处。
- 单个资源有时表示来自多个底层（最终是一致的）数据源的数据的合并。在这些情况下，不可能返回未更改字段的权威信息。

在这种情况下，API <font color="#188038">可以</font>只返回已更新的字段，而忽略其余字段，如果这样做，则<font color="#f57c00">应该</font>记录此行为。

### 错误

请参阅[errors](#AIP-193)，特别是何时使用 `PERMISSION_DENIED` 和 `NOT_FOUND` 错误。

此外，如果用户确实具有适当的权限，但所请求的资源不存在，则除非 `allow_missing` 设置为 `true`，否则服务<font color="#c5221f">必须</font>出现 `NOT_FOUND`（HTTP 404）错误

### Changelog

- 2022-11-04: Aggregated error guidance to AIP-193.
- 2022-06-02: Changed suffix descriptions to eliminate superfluous "-".
- 2021-11-04: Changed the permission check if allow_missing is set.
- 2021-07-08: Added error guidance for resource not found case.
- 2021-03-05: Changed the etag error from FAILED_PRECONDITION (which becomes HTTP 400) to ABORTED (409).
- 2020-10-06: Added guidance for declarative-friendly resources.
- 2020-10-06: Added guidance for allow_missing.
- 2020-08-14: Added error guidance for permission denied cases.
- 2020-06-08: Added guidance on returning the full resource.
- 2019-10-18: Added guidance on annotations.
- 2019-09-10: Added a link to the long-running operations AIP.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to present a better example of resource ownership.
- 2019-06-10: Added guidance for long-running update.
- 2019-05-29: Added an explicit prohibition on arbitrary fields in standard methods.

## **AIP-135** 标准方法：删除（Standard methods: Delete） {#AIP-135}

{{< admonition info >}}
翻译时间：2023-04-06
{{< /admonition >}}

在 REST API 中，通常对资源的 URI（例如 `/v1/publisher/{publisher}/books/{book}`）发出 `DELETE` 请求以删除该资源。

面向资源的设计（[AIP-121](#AIP-121)）通过 `Delete` 方法来尊重这种模式。这些 RPC 接受表示该资源的 URI，并且通常返回一个空响应。

### 指导

API 通常<font color="#f57c00">应该</font>为资源提供一个删除方法，除非这样做对用户来说没有价值。

Delete 方法是使用以下模式指定的：

```protobuf
rpc DeleteBook(DeleteBookRequest) returns (google.protobuf.Empty) {
  option (google.api.http) = {
    delete: "/v1/{name=publishers/*/books/*}"
  };
  option (google.api.method_signature) = "name";
}
```

- RPC 的名称<font color="#c5221f">必须</font>以 `Delete` 一词开头。RPC 名称的其余部分<font color="#f57c00">应该</font>是资源消息名称的单数形式。
- 请求消息<font color="#c5221f">必须</font>与 RPC 名称匹配，并带有 `Request` 后缀。
- 响应消息<font color="#f57c00">应该</font>是 `google.protobuf.Empty`。
  - 如果资源被[软删除](#AIP-135)，那么响应消息<font color="#f57c00">应该</font>是资源本身。
  - 如果删除 RPC 是[长时间运行的](#AIP-134)，则响应消息<font color="#c5221f">必须</font>是解析为正确响应的 `google.longrunning.Operation`。
- HTTP 谓词<font color="#c5221f">必须</font>是 `DELETE`。
- 接收资源名称的请求消息字段<font color="#f57c00">应该</font>映射到 URI 路径。
  - 此字段应称为 `name`。
  - `name` 字段<font color="#f57c00">应该</font>是 URI 路径中唯一的变量。所有剩余的参数都<font color="#f57c00">应该</font>映射到 URI 查询参数。
- `google.api.http` 注释中<font color="#c5221f">不能</font>有 `body` 键。
- <font color="#f57c00">应该</font>只有一个值为 `"name"` 的 `google.api.method_signature` 注释。如果使用了 `etag` 或 `force field`，它们<font color="#188038">可以</font>包含在签名中。

如果并且仅当资源存在并且已成功删除时，Delete 方法<font color="#f57c00">应</font>成功。如果资源不存在，则该方法<font color="#f57c00">应</font>返回 `NOT_FOUND` 错误。

### 请求消息

Delete 方法实现了一种常见的请求消息模式：

```protobuf
message DeleteBookRequest {
  // The name of the book to delete.
  // Format: publishers/{publisher}/books/{book}
  string name = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      type: "library.googleapis.com/Book"
    }];
}
```

- <font color="#c5221f">必须</font>包含 `name` 字段。它<font color="#f57c00">应该</font>被称为 `name`。
  - <font color="#f57c00">应</font>根据需要对字段进行[注释](#AIP-203)。
  - 字段<font color="#c5221f">必须</font>标识其引用的[资源类型](#AIP-123)。
- 字段的注释<font color="#f57c00">应该</font>记录资源模式。
- 请求消息<font color="#c5221f">不得</font>包含任何其他必需字段，也<font color="#f57c00">不应</font>包含除本 AIP 或其他 AIP 中描述的字段之外的其他可选字段。

### 软删除

{{< admonition note >}}
该材料被移到自己的文件中，以提供更全面的处理：[AIP-164](#AIP-164)。
{{< /admonition >}}

### 长时间运行的删除（Long-running delete）

某些资源删除资源所需的时间比常规 API 请求所需的合理时间要长。在这种情况下，API <font color="#f57c00">应该</font>使用长时间运行的操作：

```protobuf
rpc DeleteBook(DeleteBookRequest) returns (google.longrunning.Operation) {
  option (google.api.http) = {
    delete: "/v1/{name=publishers/*/books/*}"
  };
  option (google.longrunning.operation_info) = {
    response_type: "google.protobuf.Empty"
    metadata_type: "OperationMetadata"
  };
}
```

- 如果 RPC 不是长时间运行的，则响应类型<font color="#c5221f">必须</font>设置为适当的返回类型：对于大多数 Delete RPC，为 `google.protobuf.Empty`；对于软删除，为资源本身（[AIP-164](#AIP-164)）。
- <font color="#c5221f">必须</font>同时指定 `response_type` 和 `metadata_type` 字段（即使它们是 `google.protobuf.Empty`）。

### 级联删除

有时，用户可能需要删除资源以及所有适用的子资源。然而，由于删除通常是永久性的，让用户不要意外地删除也很重要，因为重建被删除的子资源可能相当困难。

如果 API 允许删除可能具有子资源的资源，则 API <font color="#f57c00">应</font>在请求上提供 `bool force` 字段，用户将其设置为明确选择级联删除。

```protobuf
message DeletePublisherRequest {
  // The name of the publisher to delete.
  // Format: publishers/{publisher}
  string name = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      type: "library.googleapis.com/Publisher"
    }];

  // If set to true, any books from this publisher will also be deleted.
  // (Otherwise, the request will only work if the publisher has no books.)
  bool force = 2;
}
```

如果 `force` 字段为 `false`（或未设置）并且存在子资源，则 API <font color="#c5221f">必须</font>失败，并返回 `FAILED_PRECONDITION` 错误。

### 受保护的删除

有时，用户可能需要确保没有对正在删除的资源进行任何更改。如果资源提供了 [etag](#AIP-134)，则删除请求<font color="#188038">可以</font>接受 etag（作为必须或可选字段）：

```protobuf
message DeleteBookRequest {
  // The name of the book to delete.
  // Format: publishers/{publisher}/books/{book}
  string name = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference) = {
      type: "library.googleapis.com/Book"
    }];

  // Optional. The etag of the book.
  // If this is provided, it must match the server's etag.
  string etag = 2;
}
```

如果提供的 etag 与服务器计算的 etag 不匹配，则请求<font color="#c5221f">必须</font>失败，并返回 `ABORTED` 错误代码。

{{< admonition note >}}
声明性友好资源（[AIP-128](#AIP-128)）<font color="#c5221f">必须</font>为 Delete 请求提供 `etag` 字段。
{{< /admonition >}}

### 删除（如果存在）

如果服务使用客户端分配的资源名称，Delete 方法<font color="#188038">可以</font>暴露一个 `bool allow_missing` 字段，这将导致该方法在用户试图删除不存在的资源时成功（在这种情况下，请求是无操作的）：

```protobuf
message DeleteBookRequest {
  // The book to delete.
  // Format: publishers/{publisher}/books/{book}
  string name = 1 [
    (google.api.field_behavior) = REQUIRED,
    (google.api.resource_reference).type = "library.googleapis.com/Book"
  ];

  // If set to true, and the book is not found, the request will succeed
  // but no action will be taken on the server
  bool allow_missing = 2;
}
```

更具体地说，`allow_missing` 标志触发以下行为：

- 如果方法调用位于不存在的资源上，则请求是无操作的。
  - `etag` 字段被忽略。
- 如果方法调用位于已存在的资源上，则会删除该资源（接受其他检查）。

{{< admonition note >}}
声明性友好资源（[AIP-128](#AIP-128)）<font color="#f57c00">应该</font>公开 `bool allow_missing` 字段。
{{< /admonition >}}

### 错误

如果用户没有访问该资源的权限，无论该资源是否存在，该服务都<font color="#c5221f">必须</font>返回 `PERMISSION_DENIED`（HTTP 403）错误。在检查资源是否存在之前，<font color="#c5221f">必须</font>检查权限。

如果用户确实具有适当的权限，但所请求的资源不存在，则除非 `allow_missing` 设置为 `true`，否则服务<font color="#c5221f">必须</font>返回 `NOT_FOUND`（HTTP 404）错误。

### 延伸阅读

- 有关软删除和取消删除，请参阅 [AIP-164](#AIP-164)。
- 有关基于筛选器批量删除资源的信息，请参阅 [AIP-165](#AIP-165)。

### Changelog

- 2022-06-02: Changed suffix descriptions to eliminate superfluous "-".
- 2022-02-02: Changed eTag error from `FAILED_PRECONDITION` to `ABORTED` making it consistent with change to [AIP-154](<(#AIP-154)>) & [AIP-134](#AIP-134) on 2021-03-05.
- 2020-10-06: Added guidance for declarative-friendly resources.
- 2020-10-06: Added guidance for allowing no-op delete for missing resources.
- 2020-10-06: Moved soft delete and undelete guidance into a new [AIP-164](#AIP-164).
- 2020-06-08: Added guidance for Get of soft-deleted resources.
- 2020-02-03: Added guidance for error cases.
- 2019-10-18: Added guidance on annotations.
- 2019-08-01: Changed the examples from "shelves" to "publishers", to present a better example of resource ownership.
- 2019-06-10: Added guidance for long-running delete.
- 2019-05-29: Added an explicit prohibition on arbitrary fields in standard methods.
