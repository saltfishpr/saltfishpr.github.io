---
title: gRPC JWT Auth
date: 2023-04-04T15:41:06Z
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: GRPC 实现 JWT 认证
tags: ["go", "grpc"]
categories: ["Programming"]
---

<!--more-->

GRPC 实现自定义认证，这里用 jwt token 作为示例。

定义认证方式，实现 `credentials.PerRPCCredentials` 接口

```go
const MetadataKeyAuth = "authorization"

// Auth 自定义认证。
type Auth struct {
	Token string
}

// GetRequestMetadata 获取认证信息。
func (c Auth) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return map[string]string{
		MetadataKeyAuth: c.Token,
	}, nil
}

// RequireTransportSecurity 是否需要安全传输。
func (c Auth) RequireTransportSecurity() bool {
	return false
}
```

### 客户端

- 多次鉴权：创建连接时添加额外的 option `grpc.WithPerRPCCredentials(auth)`，可以编写一个 TokenManager 用于缓存/刷新 token

  ```go
  	tokenStr, err := jwt.Generate("saltfish")
  	if err != nil {
  		log.Fatalf("generate token failed: %s", err)
  	}
  	auth := jwt.Auth{Token: tokenStr}

  	conn, err := grpc.Dial(":9000", grpc.WithInsecure(), grpc.WithPerRPCCredentials(auth))
  	if err != nil {
  		log.Fatalf("did not connect: %s", err)
  	}
  	c := userv1.NewUserServiceClient(conn)

  	r, err := c.CreateUser(context.Background(), &userv1.CreateUserRequest{})
  ```

- 单次鉴权：请求时添加额外的 option `grpc.PerRPCCredentials(auth)`

  ```go
  	conn, err := grpc.Dial(":9000", grpc.WithInsecure())
  	if err != nil {
  		log.Fatalf("did not connect: %s", err)
  	}
  	c := userv1.NewUserServiceClient(conn)

  	tokenStr, err := jwt.Generate("saltfish")
  	if err != nil {
  		log.Fatalf("generate token failed: %s", err)
  	}
  	auth := jwt.Auth{Token: tokenStr}
  	r, err := c.CreateUser(context.Background(), &userv1.CreateUserRequest{}, grpc.PerRPCCredentials(auth))
  ```

### 服务端

使用 [go-grpc-middleware](github.com/grpc-ecosystem/go-grpc-middleware) 的 auth 中间件 `grpc_auth.UnaryServerInterceptor`，并自定义校验函数

```go
import grpc_auth "github.com/grpc-ecosystem/go-grpc-middleware/auth"

// ...
	s := grpc.NewServer(
		grpc.StreamInterceptor(grpc_middleware.ChainStreamServer(
			grpc_ctxtags.StreamServerInterceptor(),
			grpc_zap.StreamServerInterceptor(logger),
			grpc_recovery.StreamServerInterceptor(),
			grpc_auth.StreamServerInterceptor(jwt.AuthFunc), // add auth func
		)),
		grpc.UnaryInterceptor(grpc_middleware.ChainUnaryServer(
			grpc_ctxtags.UnaryServerInterceptor(),
			grpc_zap.UnaryServerInterceptor(logger),
			grpc_recovery.UnaryServerInterceptor(),
			grpc_auth.UnaryServerInterceptor(jwt.AuthFunc), // add auth func
		)),
	)
// ...
```

校验函数先从 context 中获取 token，再调用 `Verify` 方法验证 token 有效性

- `NewContext(ctx, claims)` 将 `claims` 存入上下文中

```go
import (
	"errors"

	"github.com/golang-jwt/jwt/v4"
	"google.golang.org/grpc/metadata"
)

func AuthFunc(ctx context.Context) (context.Context, error) {
	tokenStr, err := getTokenFromContext(ctx)
	if err != nil {
		return nil, err
	}

	tokenStr = strings.TrimPrefix(tokenStr, "Bearer ")
	claims, err := Verify(tokenStr)
	if err != nil {
		return nil, err
	}

	return NewContext(ctx, claims), nil
}

func getTokenFromContext(ctx context.Context) (string, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return "", errors.New("no metadata in context")
	}

	values := md[MetadataKeyAuth]
	if len(values) == 0 {
		return "", errors.New("no authorization")
	}

	return values[0], nil
}

func Verify(tokenStr string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{},
		func(token *jwt.Token) (interface{}, error) {
			return []byte(secretKey), nil
		},
	)
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, ErrorInvalidToken
	}
	return token.Claims.(*Claims), nil
}
```
