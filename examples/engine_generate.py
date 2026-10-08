from tokenrouter.entrypoints.engine import TokenRoutingEngine


def main() -> None:
    config_path = "config/r2r/Qwen3-0.6B_Qwen3-32B.yaml"

    with TokenRoutingEngine(config_path) as engine:
        response = engine.generate(
            prompt="'<|im_start|>user\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her $4$ hours, including $t$ minutes spent in the coffee shop. When she walks at $s + 2$ kilometers per hour, the walk takes her $2$ hours and $24$ minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s + \\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n<|im_end|>\n<|im_start|>assistant\n'",
            sampling_params={
                "temperature": 0,
                "top_p": 1,
                "max_new_tokens": 8192,
            },
        )

    print(response.get("text", response))
    print("meta_info:", response.get("meta_info"))


if __name__ == "__main__":
    main()
