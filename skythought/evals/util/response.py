from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Response:
    """Aggregate response information from a model backend."""

    # Text for each generated sample
    response: List[str]
    # Token counts for the corresponding completions
    num_completion_tokens: List[int]
    # Number of tokens in the input prompt
    num_input_tokens: int
    # Optional reasoning content provided by the backend, one entry per sample
    reasoning_content: Optional[List[str]] = None
    # Optional index of the request within the dataset
    index: Optional[int] = None

    @classmethod
    def from_ray_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a rayllm response.

        Args:
            response: Ray response object containing generated text and token information

        Returns:
            Responses: New instance initialized with Ray response data
        """

        if isinstance(response["generated_text"], list):
            # n > 1 samples
            response_texts = response["generated_text"]
            num_completion_tokens = [
                int(response["num_generated_tokens"][i])
                for i in range(len(response["num_generated_tokens"]))
            ]
        else:
            response_texts = [response["generated_text"]]
            num_completion_tokens = [int(response["num_generated_tokens"])]
        return cls(
            response=response_texts,
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=int(response["num_input_tokens"]),
            index=response["index"],
        )

    @classmethod
    def from_openai_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from an OpenAI response.

        Args:
            response: OpenAI response object containing message content and token information

        Returns:
            Responses: New instance initialized with OpenAI response data
        """
        return cls(
            response=[
                response.choices[i].message.content
                for i in range(len(response.choices))
            ],
            num_completion_tokens=[
                response.usage.completion_tokens if i == 0 else 0
                for i in range(len(response.choices))
            ],
            num_input_tokens=response.usage.prompt_tokens,
            reasoning_content=[
                getattr(response.choices[i].message, "reasoning_content", None)
                for i in range(len(response.choices))
            ],
        )

    @classmethod
    def from_vllm_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a vLLM response.

        Args:
            response: vLLM response object containing output text and token information

        Returns:
            Responses: New instance initialized with vLLM response data
        """
        response_texts = [
            response.outputs[i].text for i in range(len(response.outputs))
        ]
        num_completion_tokens = [
            len(response.outputs[i].token_ids) for i in range(len(response.outputs))
        ]
        return cls(
            response=response_texts,
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=len(response.prompt_token_ids),
        )


@dataclass
class SingleParsedResponse:
    content: str
    correctness: Optional[bool] = None
    reason: Optional[str] = None
    reasoning_content: Optional[str] = None

    def to_dict(self):
        return {
            "content": self.content,
            "correctness": self.correctness,
            "reason": self.reason,
            "reasoning_content": self.reasoning_content,
        }
