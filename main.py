import asyncio
import logging
import json
from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from browser_use.agent.views import AgentHistoryList
from langchain.schema import AIMessage, ChatGeneration, ChatResult
from langchain_ollama import ChatOllama

MODEL_NAME = "deepseek-r1:8b"
LLM_PARSE_COUNT = 5

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomChatOllama(ChatOllama):
    async def process_response(self, response: str) -> str:
        """ Extract and format JSON response """
        parts = response.split("```json", 1)
        if len(parts) < 2:
            parts = response.split("</think>", 1)
        
        if len(parts) < 2:
            json_str = response
        else:
            json_part = parts[1].strip()
            json_str = json_part.replace("```json", "").replace("```", "").strip()

        try:
            json_res = json.loads(json_str)
            formatted_json = json.dumps(json_res, indent=4)  # Properly format JSON
            return formatted_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}\njson_str: {json_str}")
            raise ValueError(f"JSON decoding failed: {e}")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """ Attempt to call the LLM, retrying once on failure """
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
                    raise  # Rethrow the error if all retries fail

                attempt += 1  # Increment retry counter
                await asyncio.sleep(1)  # Small delay before retry

async def run_search() -> AgentHistoryList:
    """ Run the browser agent to search for cat pictures """
    try:
        context_config = BrowserContextConfig(
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

        browser = Browser()
        context = BrowserContext(browser=browser, config=context_config)
        initial_actions = [{"open_tab": {"url": "https://www.google.com"}}]

        model = CustomChatOllama(
            model=MODEL_NAME,
            num_ctx=32000,
            streaming=True,
        )

        agent = Agent(
            initial_actions=initial_actions,
            browser=browser,
            browser_context=context,
            use_vision=False,
            task="search google for cat pictures",
            llm=model,
            planner_llm=model,
            use_vision_for_planner=False,
            save_conversation_path="logs/conversation/history.json",
            max_failures=5,
        )

        result = await agent.run()
        logger.debug("Agent executed actions: %s", result.action_names())
        logger.debug("Extracted content: %s", result.extracted_content())
        logger.debug("Errors (if any): %s", result.errors())
        logger.debug("Model actions: %s", result.model_actions())
        return result

    except Exception as e:
        logger.error("Error in run_search: %s", e, exc_info=True)
        raise

async def main():
    result = await run_search()
    logger.debug("Final Result: %s", result)
    
    if hasattr(result, "steps"):
        logger.debug("Agent History Steps:")
        for step in result.steps:
            logger.debug("Step: %s, details: %s", step.name, step.details)
            if hasattr(step, "raw_output"):
                logger.debug("Raw Output: %s", step.raw_output)
    else:
        logger.debug("No detailed history available.")

if __name__ == "__main__":
    asyncio.run(main())
