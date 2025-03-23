import asyncio
import os
import dotenv

dotenv.load_dotenv()

import sys

airtrain_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(airtrain_path)

from airtrain.integrations.openai.skills import OpenAIChatSkill, OpenAIInput


async def main():
    skill = OpenAIChatSkill()

    # Example 1: Basic async chat
    input_data = OpenAIInput(
        user_input="Explain quantum computing in simple terms",
        system_prompt="You are a physics professor",
        model="gpt-4o",
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
