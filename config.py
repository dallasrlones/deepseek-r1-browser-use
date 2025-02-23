import json
import re

from browser_use.browser.context import BrowserContextConfig, BrowserContext
from langchain.schema import AIMessage, ChatGeneration, ChatResult
from langchain_ollama import ChatOllama

import logging
from browser_use import Browser

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LLM_PARSE_COUNT = 5

CONTEXT_CONFIG = BrowserContextConfig(
    cookies_file="path/to/cookies.json",
    wait_for_network_idle_page_load_time=3.0,
    browser_window_size={"width": 1280, "height": 1100},
    locale="en-US",
    user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/85.0.4183.102 Safari/537.36"
    ),
    highlight_elements=True,
    viewport_expansion=500,
)

class CustomChatOllama(ChatOllama):
    async def process_response(self, response: str) -> str:
        json_pattern = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL) or \
                    re.search(r"</think>\s*(\{.*?\})", response, re.DOTALL)

        json_str = json_pattern.group(1) if json_pattern else response.strip()

        try:
            json_res = json.loads(json_str)
            return json.dumps(json_res, indent=4)  # Properly format JSON
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}\njson_str: {json_str}")
            raise ValueError(f"JSON decoding failed: {e}")
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        max_retries = LLM_PARSE_COUNT
        attempt = 0

        while attempt <= max_retries:
            try:
                logger.info(f"Attempt {attempt + 1}: Calling LLM...")
                response_message = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

                chat_generation_content = response_message.generations[0].text
                processed = await self.process_response(chat_generation_content)

                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=processed)
                        )
                    ]
                )
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")

                if attempt == max_retries:
                    logger.error("Max retry limit reached. Raising exception.")
                    raise

                attempt += 1

BROWSER = Browser()
CONTEXT = BrowserContext(browser=BROWSER, config=CONTEXT_CONFIG)