---
title: Factorio (异星工厂)
date: 2024-08-07T22:51:11+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
categories: ["Game"]

draft: true
---

<!--more-->

## 启动服务

```shell
docker run -d -it \
	-p 34197:34197/udp \
	-p 27015:27015/tcp \
	-v /opt/factorio:/factorio \
	-e LOAD_LATEST_SAVE=false \
	-e SAVE_NAME=saltfishpr \
	--name factorio \
	--restart=always \
	factoriotools/factorio
```

## 手变长
/c local reach = 100
game.player.force.character_build_distance_bonus = reach
game.player.force.character_reach_distance_bonus = reach

## 物品栏
/c game.player.force.character_inventory_slots_bonus = 10

## 建筑机器人速度
/c game.player.force.worker_robots_speed_modifier = 5
/c game.player.force.worker_robots_battery_modifier  = 0
/c game.player.force.worker_robots_storage_bonus  = 5
