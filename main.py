import asyncio
import logging
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from config import BROWSER, CustomChatOllama, CONTEXT

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-r1:8b"

MODEL = CustomChatOllama(
    model=MODEL_NAME,
    num_ctx=32000,
    streaming=True
)

async def run_search(task_str, max_failures=5) -> AgentHistoryList:
    try:
        agent = Agent(
            initial_actions=[{"open_tab": {"url": "https://www.google.com"}}],
            browser=BROWSER,
            browser_context=CONTEXT,
            use_vision=False,
            task=task_str,
            llm=MODEL,
            planner_llm=MODEL,
            use_vision_for_planner=False,
            save_conversation_path="logs/conversation/history.json",
            max_failures=max_failures
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
    result = await run_search("search google for cat pictures", max_failures=100)
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
