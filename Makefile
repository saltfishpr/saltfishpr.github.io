.PHONY: post dev help

POST_NAME ?=

post:
ifndef POST_NAME
	$(error POST_NAME is not set. Usage: make mkpost POST_NAME="your-post-name")
endif
	@hugo new "posts/$(date +"%Y-%m-%d")-$(POST_NAME)/index.zh-cn.md"

dev:
	@hugo server -D
