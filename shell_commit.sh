#!/usr/bin/env bash

git pull
git add -A
git commit -m "$(date "+%Y-%m-%d %H:%M:%S")"
git push
