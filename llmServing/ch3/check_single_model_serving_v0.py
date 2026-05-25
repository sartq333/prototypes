from single_model_serving_v0.llm import LLMEngine

if __name__ == "__main__":
    engine = LLMEngine()
    results = engine.generate(
        ["What is the capital of France", "Who is Narendra Modi?"]
    )