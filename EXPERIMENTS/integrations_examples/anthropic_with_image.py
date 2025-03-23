import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my_api_key",
)

# Replace placeholders like {{PR_DESCRIPTION}} with real values,
# because the SDK does not support variables.
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=8192,
    temperature=0,
    system="You are an experienced software engineer tasked with reviewing a GitHub Pull Request (PR). Your goal is to analyze the code quality and suggest improvements. Follow these steps carefully:\n\n1. Review the PR description:\n<PR_DESCRIPTION>\n{{PR_DESCRIPTION}}\n</PR_DESCRIPTION>\n\n2. Examine the code changes:\n<CODE_CHANGES>\n{{CODE_CHANGES}}\n</CODE_CHANGES>\n\n3. Consider any existing comments:\n<EXISTING_COMMENTS>\n{{EXISTING_COMMENTS}}\n</EXISTING_COMMENTS>\n\n4. Analyze the code quality:\n   a. Check for adherence to coding standards and best practices\n   b. Evaluate code readability and maintainability\n   c. Assess performance implications\n   d. Look for potential bugs or edge cases\n   e. Consider security implications\n\n5. Suggest improvements:\n   a. Identify areas where the code can be optimized or simplified\n   b. Propose alternative approaches if applicable\n   c. Recommend additional tests or error handling if needed\n\n6. Format your response as follows:\n   <code_review>\n   <quality_analysis>\n   Provide a detailed analysis of the code quality, addressing points 4a-4e.\n   </quality_analysis>\n\n   <improvement_suggestions>\n   List your suggestions for improvement, addressing points 5a-5c. Number each suggestion.\n   </improvement_suggestions>\n\n   <summary>\n   Provide a brief summary of your overall assessment and key recommendations.\n   </summary>\n   </code_review>\n\nRemember to be constructive and specific in your feedback. Use code snippets or pseudocode to illustrate your suggestions when appropriate. If you need clarification on any part of the code or PR description, state your assumptions clearly.\n\nDo not comment on aspects unrelated to code quality or potential improvements. Focus solely on the technical aspects of the code changes presented.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\nAnalyze the above examples and give me some updates. Analyze this image as well.\n\nOne more image is this. Can you image this as well.",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "<base64_encoded_image>",
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "<base64_encoded_image>",
                    },
                },
            ],
        }
    ],
)
print(message.content)
