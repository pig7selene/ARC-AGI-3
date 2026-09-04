# Kaggle packaging

Keep the `arc3` package and model weights in the Kaggle dataset/notebook. Construct `arc_agi.Arcade(operation_mode=OperationMode.COMPETITION)`, make each environment once, wrap it with `adapt_environment`, load a local OpenAI-compatible (for example vLLM) model, and use the official toolkit scorer/artifact writer. Competition Mode disallows game resets and scores all environments. Internet must remain disabled during evaluation. A notebook is intentionally not generated until the current competition API is available locally, because its contract changes between toolkit releases.

`submission.py` is a starter entry point for an offline image. It intentionally fails fast when the toolkit's public registry is absent and does not fabricate a submission format.
