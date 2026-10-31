from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from langchain.tools import tool

load_dotenv()  # Replace with your actual Tavily API key

MAX_ITERATIONS = 10
OLLAMA_MODEL = "ollama:qwen3.5:0.8b"
GEMINI_MODEL = "gemini-3.8-flash"

prices = {
    "laptop": 1000.0,
    "keyboard": 200.0,
    "headphones": 150.0,
}


@tool
def get_products_list() -> list:
    """_summary_

    Returns:
        list: List of available products
    """

    return list(prices)


# tools
@tool
def get_product_price(product: str) -> float:
    """Get the price of a product.

    Args:
        product (str): The name of the product.

    Returns:
        float: The price of the product.
    """
    print(f"Getting price for product: {product}")
    prices = {
        "laptop": 1000.0,
        "keyboard": 200.0,
        "headphones": 150.0,
    }
    return prices.get(product, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return final price

    Args:
        price (float): Price to apply discount for
        discount_tier (str): Discount tier, can be one of "bronze", "silver", "gold"

    Returns:
        float: Final price after discount
    """
    print(f"Applying discount of tier {discount_tier} to price: {price}")
    discount_tiers = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_tiers.get(discount_tier, 0)
    return round(price * (100 - discount) / 100, 2)


# agent loop
@traceable(name="agent_loop_langchain_tool_calling", tags={"course": "langchain"})
def run_agent(question: str):
    tools = [get_products_list, get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    model = GEMINI_MODEL
    llm = init_chat_model(model, model_provider="google_genai", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content=("""
            You are a helpful shopping assitant and have access to a product catalog tool 
            and discount tool. 
            You can use get_products_list to get the list of products available
            STRICT RULES - You must call get_product_price to get the product price. 
            and only call apply_discount AFTER you have receibed a price from get_product_price. 
            Pass the exact price, do NOT pass a made up number.
            NEVER calculate discounts by yourself, always use the apply_discount tool.
            """)),
        HumanMessage(content=(question)),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--ITERATION {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            final_answer = (
                ai_message.content[0]["text"]
                if model == GEMINI_MODEL
                else ai_message.content
            )
            print(f"Final answer: {final_answer}")
            return final_answer

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise (ValueError(f"Tool '{tool_name} not found"))

        tool_result = tool_to_use.invoke(tool_args)

        print(" [TOOL RESULT]: " + str(tool_result))
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
        )

    print("Error: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello from langchain-course!")
    result = run_agent(
        "What is the price of a laptop after applying a silver discount?"
    )
