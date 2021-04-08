#!/bin/bash
cp processed/encoded_games_detailed_info.fethear models
zip -r models-$(date +%Y-%m-%d:%H:%M) models
