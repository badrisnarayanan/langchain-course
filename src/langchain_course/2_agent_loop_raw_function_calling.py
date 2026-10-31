from dotenv import load_dotenv
from langsmith import traceable
import ollama

load_dotenv()  # Replace with your actual Tavily API key

MAX_ITERATIONS = 10
OLLAMA_MODEL = "qwen3.5:0.8b"

prices = {
    "laptop": 1000.0,
    "keyboard": 200.0,
    "headphones": 150.0,
}


# tools
@traceable(run_type="tool")
def get_products_list() -> list:
    """Gives list of all products available

    Returns:
        list: List of available products
    """

    return list(prices)


@traceable(run_type="tool")
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


@traceable(run_type="tool")
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


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_products_list",
            "description": "Gives List of all available products",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=OLLAMA_MODEL, tools=tools_for_llm, messages=messages)


# agent loop
@traceable(name="ollama_agent_loop", tags={"course": "langchain"})
def run_agent(question: str):
    tools_dict = {
        "get_products_list": get_products_list,
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--ITERATION {iteration} ---")

        response = ollama_chat_traced(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            final_answer = ai_message.content
            print(f"Final answer: {final_answer}")
            return final_answer

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise (ValueError(f"Tool '{tool_name} not found"))

        tool_result = tool_to_use(**tool_args)

        print(" [TOOL RESULT]: " + str(tool_result))
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(tool_result),
            }
        )

    print("Error: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello from langchain-course!")
    result = run_agent(
        "What is the price of a laptop after applying a silver discount?"
    )
