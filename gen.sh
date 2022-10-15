#!/bin/sh

pip3 install -r requirements.txt

for f in gen/*.py
    do python3 "$f"
done

