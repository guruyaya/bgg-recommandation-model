#!/bin/bash
cp processed/encoded_games_detailed_info.fethear models
cp config.yml models
try_model.py models/saved_model.json
zip -r models-$(date +%Y-%m-%d:%H:%M) models
