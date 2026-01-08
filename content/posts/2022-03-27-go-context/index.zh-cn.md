---
title: Go context
date: 2022-03-27T00:34:33+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: context 包主要用来在 goroutine 之间传递上下文信息，包括：取消信号、超时时间、截止时间、k-v 等。提供对 goroutine 的并发控制和超时控制。
tags: ["go"]
categories: ["Programming"]
---

<!--more-->

花了两个晚上学习了一下 Go context 包，顺便记录下来。

context 包主要用来在 goroutine 之间传递上下文信息，包括：取消信号、超时时间、截止时间、k-v 等。提供对 goroutine 的并发控制和超时控制。

```shell
go version
```

```text
go version go1.18 darwin/arm64
```

## 源码阅读

### Context 接口

Context 定义了 4 个方法，它们都是幂等的。

- 取消某个 Context 时，从该 Context 派生的所有 Context 也将被取消。
- WithCancel、WithDeadline 和 WithTimeout 函数传入父 Context 并返回派生出的子 Context 和 CancelFunc。
- 调用 CancelFunc 会取消子节点和子节点的子节点，删除父节点对子节点的引用，并停止任何关联的计时器。
- 调用 CancelFunc 失败会泄漏子节点和子节点的子节点，直到父节点被取消或计时器触发。
- go vet 会工具检查是否在所有控制流路径上使用了 CancelFuncs。

```go
type Context interface {
	// Deadline 返回代表此 Context 完成的工作应被取消的时间。
	// 未设置截止时间时，Deadline 返回 ok==false。
	// 连续调用 Deadline 返回相同的结果。
	Deadline() (deadline time.Time, ok bool)
	// 当工作完成，Done 会返回一个关闭的通道，说明此 Context 应该被取消。
	// 如果这个 Context 永远不能被取消，Done 可能会返回 nil。
	// 连续调用 Done 返回相同的结果。
	// 在 cancel() 函数返回之后，通道的关闭可能会异步进行。
	Done() <-chan struct{}
	// 如果 Done 尚未关闭，则 Err 返回 nil。
	// 如果 Done 已关闭，Err 将返回一个 non-nil 的错误来解释原因：
	//   Cancel: Context 被取消
	//   DeadlineExceeded: Context 超出截止时间
	Err() error
	// Value 为 key 返回与此 Context 关联的值，如果没有值与 key 关联，则返回 nil。
	// 使用相同的键连续调用 Value 将返回相同的结果。
	// Context 值仅用于传输进程和 API 边界的请求范围数据，而不用于向函数传递可选参数。
	// key 标识 Context 中的特定值。希望在 Context 中存储值的函数通常会在全局变量中分配一个键，然后将该键用作 context.WithValue 和 Context.Value 的参数。
	// key 可以是任何支持比较的类型；包应将键定义为未导出的类型以避免冲突。
	Value(key any) any
}
```

### emptyCtx

emptyCtx 实现一个空的 Context，永远不会被取消，没有存储值，也没有 deadline。

```go
type emptyCtx int

func (*emptyCtx) Deadline() (deadline time.Time, ok bool) {
	return
}

func (*emptyCtx) Done() <-chan struct{} {
	return nil
}

func (*emptyCtx) Err() error {
	return nil
}

func (*emptyCtx) Value(key any) any {
	return nil
}

func (e *emptyCtx) String() string {
	switch e {
	case background:
		return "context.Background"
	case todo:
		return "context.TODO"
	}
	return "unknown empty Context"
}
```

context 包将 emptyCtx 包装成两个常用空 Context，并通过函数导出

```go
var (
	background = new(emptyCtx)
	todo       = new(emptyCtx)
)

func Background() Context {
	return background
}

func TODO() Context {
	return todo
}
```

background 通常用在 main 函数中，作为所有 Context 的根节点。

todo 通常使用在不知道传什么 Context 的场景，如代码重构，用于占个位置。

### cancelCtx

cancelCtx 是 context 包的核心，它提供了一个可取消的上下文。

```go
var closedchan = make(chan struct{})

func init() {
	close(closedchan) // 创建一个可重用的，关闭状态的通道
}

// &cancelCtxKey 是返回 cancelCtx 自身的特殊 key。
var cancelCtxKey int

type cancelCtx struct {
	Context

	mu       sync.Mutex            // 保护以下字段
	done     atomic.Value          // 保存 chan struct{}，懒汉式创建，调用一次 cancel() 时关闭
	children map[canceler]struct{} // 在第一次调用 cancel() 时设置为 nil
	err      error                 // 在第一次调用 cancel() 时设置为 non-nil
}

func (c *cancelCtx) Value(key any) any {
	// 如果 key 等于 &cancelCtxKey，就返回该 cancelCtx 本身
	if key == &cancelCtxKey {
		return c
	}
	// 不相等，向上递归查找有没有匹配该 key 的 value，这里传入 parent
	return value(c.Context, key)
}

func value(c Context, key any) any {
	for {
		switch ctx := c.(type) {
		case *valueCtx:
			if key == ctx.key {
				return ctx.val // 找到了对应的 key
			}
			c = ctx.Context // 向上寻找
		case *cancelCtx:
			if key == &cancelCtxKey {
				return c // 如果 key 为 &cancelCtxKey 就返回该 cancelCtx 自身
			}
			c = ctx.Context
		case *timerCtx:
			if key == &cancelCtxKey {
				return &ctx.cancelCtx // 由于 timerCtx 内部包含一个 cancelCtx，直接返回内部的 cancelCtx
			}
			c = ctx.Context
		case *emptyCtx:
			return nil // emptyCtx 不含 key-val 对
		default:
			return c.Value(key) // 递归向上寻找
		}
	}
}

func (c *cancelCtx) Done() <-chan struct{} {
	// 懒汉式加载
	// 如果 d.done 不为空，直接返回
	// 如果 d.done 为空，make 一个 chan struct{} 存入 c.done 中并返回
	// 即 c.Done() 至少被调用一次，c.done 才不为空
}

func (c *cancelCtx) Err() error {
	// ...
}

func (c *cancelCtx) String() string {
	return contextName(c.Context) + ".WithCancel" // 父节点名称.WithCancel
}

// cancel 关闭 c.done，取消 c 的每个子节点，如果 removeFromParent 为真，从其父亲的子节点中删除 c。
func (c *cancelCtx) cancel(removeFromParent bool, err error) {
	// 必须传入 err
	if err == nil {
		panic("context: internal error: missing cancel error")
	}
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return // 已经被其他 goroutine 取消
	}
	c.err = err // 给 err 赋值，注意此时还在 lock 状态
	d, _ := c.done.Load().(chan struct{})
	if d == nil {
		c.done.Store(closedchan) // 没有初始化过 done channel，就直接赋值一个已关闭的 chan
	} else {
		close(d)
	}
	for child := range c.children {
		// 递归取消所有子节点
		// 注意：该节点的子节点并不从该节点移除
		// 注意：在持有父锁的同时获取子锁。
		child.cancel(false, err)
	}
	// 置空子节点
	c.children = nil
	c.mu.Unlock()

	if removeFromParent {
		// 从父节点中移除自己
		removeChild(c.Context, c)
	}
}
```

创建可取消的 Context 的方法

```go
// CancelFunc 告诉操作放弃它的工作。
// CancelFunc 不会等待工作停止。
// 一个 CancelFunc 可以被多个 goroutine 同时调用。
// 在第一次调用之后，对 CancelFunc 的后续调用什么也不做。
type CancelFunc func()

// WithCancel 返回具有新 Done 通道的 parent 副本。
// 返回的 Context 的 Done 通道在调用返回的 cancel 函数或父 Context 的 Done 通道关闭时关闭，以先发生者为准。
// 取消此 Context 会释放与其关联的资源，因此代码应在此 Context 中运行的操作完成后立即调用取消。
func WithCancel(parent Context) (ctx Context, cancel CancelFunc) {
	if parent == nil {
		panic("cannot create context from nil parent")
	}
	c := newCancelCtx(parent)
	propagateCancel(parent, &c)
	return &c, func() { c.cancel(true, Canceled) }
}

// newCancelCtx 返回一个初始化的 cancelCtx。
func newCancelCtx(parent Context) cancelCtx {
	return cancelCtx{Context: parent}
}


// goroutines 计算曾经创建的 goroutines 的数量；供测试用。
var goroutines int32

// propagateCancel 构建父节点与子节点之间的关联关系，在 parent 被取消时取消 child 及其子节点。
func propagateCancel(parent Context, child canceler) {
	done := parent.Done()
	if done == nil {
		return // parent 永远不会被取消
	}

	// 非阻塞判断 parent 是否被取消
	select {
	case <-done:
		// parent 已被取消，取消 child 及其子节点后返回
		child.cancel(false, parent.Err())
		return
	default:
	}

	// 寻找 parent 的可以取消的父节点
	if p, ok := parentCancelCtx(parent); ok {
		p.mu.Lock()
		if p.err != nil {
			// parent 已被取消，取消 child 及其子节点
			child.cancel(false, p.err)
		} else {
			// 将 child 存入 parent 的 map 中
			if p.children == nil {
				p.children = make(map[canceler]struct{})
			}
			p.children[child] = struct{}{}
		}
		p.mu.Unlock()
	} else {
		atomic.AddInt32(&goroutines, +1)
		// 没有找到可取消的父节点，启动一个 goroutine 监听 parent 的结束信号
		go func() {
			select {
			case <-parent.Done():
				child.cancel(false, parent.Err())
			case <-child.Done():
			}
		}()
	}
}

// parentCancelCtx 返回 parent 的第一个祖先 cancelCtx 节点。
// 它通过查找 parent.Value(&cancelCtxKey) 来找到最里面的封闭 *cancelCtx，然后检查 parent.Done() 是否与 *cancelCtx 匹配。
// （如果没有，*cancelCtx 已经被包装在一个自定义实现中，提供了一个不同的完成通道，在这种情况下我们不应该绕过它。）
func parentCancelCtx(parent Context) (*cancelCtx, bool) {
	done := parent.Done()
	// parent 已经被取消或者无法永远不会被取消，直接返回
	if done == closedchan || done == nil {
		return nil, false
	}
	// 向上递归寻找最近的 cancelCtx。由上面的 cancelCtx.Value() 可知，当传入参数为 &cancelCtxKey 时，返回 c 自身。否则向上寻找。
	p, ok := parent.Value(&cancelCtxKey).(*cancelCtx)
	if !ok {
		return nil, false
	}
	pdone, _ := p.done.Load().(chan struct{})
	// 当前 parent 类型不是标准 cancelCtx 时，返回 false
	if pdone != done {
		return nil, false
	}
	return p, true // 返回找到的 cancelCtx
}
```

parentCancelCtx 由 parent 向上寻找最近的 cancelCtx，这个 cancelCtx 可以是 parent 本身。当这个 _cancelCtx_ 的 Done() 方法被重写，该 _cancelCtx_ 就不是标准的 cancelCtx，context 就无法保证所有通过 done channel 通知的 goroutine 被正确的关闭。例如：

```go
package main

import (
	"context"
)

type MyContext struct {
	context.Context
}

// Done 重写 cancelCtx 的 Done 方法
func (c MyContext) Done() <-chan struct{} {
	return make(<-chan struct{})
}

func main() {
	ctx := context.Background()            // ctx: emptyCtx
	c1, cancel1 := context.WithCancel(ctx) // c1: cancelCtx{Context: ctx, done: nil, children: nil, err: nil}
	defer cancel1()
	c2 := MyContext{Context: c1}          // c2: MyContext{Context: c1}
	c3, cancel2 := context.WithCancel(c2) // c3: cancelCtx{Context: c2, done: chan struct{}, nil, err: nil}
	defer cancel2()
	// 停在这里
	_ = c3
}
```

上面这个例子中，c1.done 为 nil，因为调用链 `WithCancel(c2)` -> `propagateCancel(c2, child)` -> `c2.Done()`，而 `c2.Done()` 被我们重写了，导致 c2.done 为 nil，在创建 c3 时 parentCancelCtx() 中 `pdone, _ := p.done.Load().(chan struct{})` 得到的 pdone 为 nil，与 `parent.Done()` 即 `c2.Done()` 返回的 `<-chan struct{}` 不相等。于是上下文在这里断开，进入 `propagateCancel()` 中的 else 分支，新开一个 goroutine 监听两个 Context 的 Done channel。

WithCancel 和 WithDeadline 返回的 cancel() 会将自身从父节点的子节点中移除

### 带取消的 Context

```go
package main

import (
	"context"
	"fmt"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // main 函数完成后主动关闭创建的 goroutine

	for n := range gen(ctx) {
		fmt.Println(n)
		if n == 5 {
			break
		}
	}
}

func gen(ctx context.Context) <-chan int {
	dst := make(chan int)
	n := 1
	go func() {
		for {
			select {
			case <-ctx.Done():
				return // 返回防止 goroutine 泄露
			case dst <- n:
				n++
			}
		}
	}()
	return dst
}
```

### timerCtx

```go
// timerCtx 带有一个计时器和一个截止时间。它嵌入了一个 cancelCtx 来实现 Done 和 Err。它通过停止计时器然后委托给 cancelCtx.cancel 来实现取消。
type timerCtx struct {
	cancelCtx
	timer *time.Timer // Under cancelCtx.mu.

	deadline time.Time
}
```

### 带 deadline 的 Context

创建时调用 `time.AfterFunc()` 方法在计时器结束时调用 `cancel()` 方法将自己从父 Context 节点中移除并通知子节点结束任务。

```go
package main

import (
	"context"
	"fmt"
	"time"
)

const shortDuration = 10 * time.Millisecond

func main() {
	d := time.Now().Add(shortDuration)
	ctx, cancel := context.WithDeadline(context.Background(), d)
	defer cancel()

	select {
	case <-time.After(1 * time.Second):
		fmt.Println("overslept")
	case <-ctx.Done():
		fmt.Println(ctx.Err())
	}
}
```

### 带 timeout 的 Context

```go
package main

import (
	"context"
	"fmt"
	"time"
)

const shortDuration = 1 * time.Millisecond

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), shortDuration)
	defer cancel()

	select {
	case <-time.After(1 * time.Second):
		fmt.Println("overslept")
	case <-ctx.Done():
		fmt.Println(ctx.Err())
	}
}
```

### valueCtx

```go
// valueCtx 带有一个键值对。它为传入的 key 实现 Value 方法并将所有其他调用委托给内部的 Context。
type valueCtx struct {
	Context
	key, val any
}
```

### 带键值对的 Context

`context.WithValue()` 创建一个带键值对的 Context，内部 Context 指向父节点。形成一个链表。

```go
package main

import (
	"context"
	"fmt"
)

func main() {
	k := "language"
	ctx := context.WithValue(context.Background(), k, "Go")

	find(ctx, k)
	find(ctx, "color")
}

func find(ctx context.Context, k string) {
	if v := ctx.Value(k); v != nil {
		fmt.Println("found value:", v)
		return
	}
	fmt.Println("key not found:", k)
}
```

## Cheat Sheet

### 请求上下文

在 Request Context 中保存整个请求都要用到的数据如 RequestID, UserID 等

TODO

## 总结

context 包加上注释不过 600 行，短小精悍，却让人眼前一亮。简单的解决了 goroutine 的控制问题，提供了主动取消的手段。

## 参考资料

[context package](https://pkg.go.dev/context)  
[深度解密 Go 语言之 context](https://www.cnblogs.com/qcrao-2018/p/11007503.html)
