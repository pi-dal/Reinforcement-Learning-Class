# Reinforcement-Learning-Class

This repository now uses Gymnasium (the actively maintained successor to OpenAI Gym).

Quick start
- Install dependencies: `pdm install` (or `pip install gymnasium` if not using PDM)
- Run the demo: `python -m demo.demo`

Notes
- API follows Gymnasium: `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`.
- The custom env is registered as `GridWorld-v0` via `gymnasium.make("GridWorld-v0", ...)`.
