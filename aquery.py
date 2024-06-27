from typing import List
import asyncio

class AsyncBatchLLMClient:
    """Create prompts combining a user query with system instructions.

    Attributes:
        model (GenerativeModel): A generative model which will be used
            to generate responses to the prompts sent to it. Must have
            a `generate_content_async` method.
    """
    def __init__(self, model: "GenerativeModel"):
        self.model = model
    
    async def aquery(
        self,
        prompts: List[str],
        batch_size: int = 30,
        sleep_time: int = 60,
    ) -> List[List[str]]:
        """Asynchronously generate responses from `self.model`
        
        A list of prompts is separated into batches, as specified by
        `batch_size`, and these are sent to `self.model` to generate
        responses. In order to comply with rate limits, you can also set
        `sleep_time` to stop sending requests for some time between each
        batch.
        
        Args:
            prompts (List[str]): The list of prompts to be sent to the
                generative model
            batch_size (int): The size of each batch
            sleep_time (int): Time to wait after sending a batch before
                sending the next batch
                
        Returns:
            results (List[str]): A list of responses for each prompt
        """
        results = []
        n = len(prompts)
        for i in range(0, n, batch_size):
            print(f"{i}/{n}    ", end="\r")
            if i >= batch_size:
                await asyncio.sleep(sleep_time)
            await self._send_batch(prompts[i:min(n, i+batch_size)], results)
        print(f"{n}/{n}    ", end="\r")
        return results
        
    async def _send_batch(self, batch: List[str], results: List[str]):
        jobs = asyncio.gather(*[self._get_response(prompt) for prompt in batch])
        results.extend(await jobs)

    async def _get_response(self, prompt: str) -> str:
        r = await self.model.generate_content_async(prompt)
        return r.text
