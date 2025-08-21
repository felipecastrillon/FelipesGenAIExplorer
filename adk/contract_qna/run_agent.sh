#!/bin/bash
source agent/.env
pip install -r requirements.txt
echo "hello" | adk run agent
