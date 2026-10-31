from dotenv import load_dotenv
from langsmith import traceable
import ollama
import re, inspect

load_dotenv()  # Replace with your actual Tavily API key

MAX_ITERATIONS = 10
OLLAMA_MODEL = "qwen3:1.7b"

prices = {
    "laptop": 1000.0,
    "keyboard": 200.0,
    "headphones": 150.0,
}


# tools
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
    price = float(price)
    print(f"Applying discount of tier {discount_tier} to price: {price}")
    discount_tiers = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_tiers.get(discount_tier, 0)
    return round(price * (100 - discount) / 100, 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


def get_tool_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # since functions are wrapped with traceable, first get original function
        original_function = (
            tool_function.__wrapped__
            if tool_function.__wrapped__ is not None
            else tool_function
        )
        # get attributes of the function
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_description(tools)
tool_names = ",".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages, options):
    return ollama.chat(model=OLLAMA_MODEL, messages=messages, options=options)


# agent loop
@traceable(name="ollama_agent_loop", tags={"course": "langchain"})
def run_agent(question: str):

    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--ITERATION {iteration} ---")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0.6, "top_p": 0.95, "num_predict": 2048},
        )
        output = response.message.content

        print(f"LLM Ouput: " + output)

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"Final answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match:
            print(" [Parsing] error, can't parse action")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            tool_result = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            tool_result = str(tools[tool_name](*args))

        print(" [TOOL RESULT]: " + str(tool_result))

        scratchpad += f"{output}\n Obsersvation: {tool_result}\n Thought:"

    print("Error: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello from langchain-course!")
    result = run_agent(
        "What is the price of a laptop after applying a silver discount?"
    )
