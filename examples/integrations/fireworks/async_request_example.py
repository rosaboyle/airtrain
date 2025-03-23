import asyncio
from airtrain.integrations.fireworks.requests_skills import (
    FireworksRequestSkill,
    FireworksRequestInput,
)


async def main():
    skill = FireworksRequestSkill()

    # Example 1: Basic async request
    input_data = FireworksRequestInput(
        user_input="Explain quantum computing in simple terms",
        system_prompt="You are a physics professor",
        model="accounts/fireworks/models/llama-v2-7b-chat",
    )

    result = await skill.process_async(input_data)
    print("Async Response:", result.response)

    # Example 2: Async streaming
    input_data.stream = True
    print("\nAsync Streaming Response:")
    async for chunk in skill.process_stream_async(input_data):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
