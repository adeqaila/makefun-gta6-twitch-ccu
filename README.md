[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adeqaila/makefun-gta6-twitch-ccu/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adeqaila/makefun-gta6-twitch-ccu/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Launch-Day Twitch CCU Forecast

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2-24) as five quantiles, from hour-1 viewers. Python stdlib only, offline, deterministic.

Model: console-midnight ratio regression with a diurnal calibration shift (+0.04 h<=5, +0.10 h6-14, +0.22 h>=15). Practice backtest mean pinball loss 0.1733 vs house baseline 0.3871.

Entry for the make.fun bounty "Forecast GTA VI's launch-day Twitch viewers hour by hour" (gta6-twitch-ccu-forecast-01).
