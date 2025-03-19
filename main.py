import re  # Regular expressions for pattern matching.
import asyncio  # For asynchronous operations.
import ollama  # For interacting with the chat model API.
from typing import Optional  # For type hinting.

# Import MCP client classes to interact with our stock API server.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import sklearn modules to implement vector search for company matching.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the parameters needed to connect to the MCP server.
server_params = StdioServerParameters(
    command="python3",           # Command to execute the Python interpreter.
    args=["stock.py"],           # The script that handles stock API requests.
    env=None,                    # No additional environment variables.
)

def extract_ticker_vector(sentence: str, company_to_ticker: dict, threshold: float = 0.3) -> Optional[str]:
    """
    Uses vector search (TF-IDF + cosine similarity) to determine the most likely company
    mentioned in the sentence and returns its ticker symbol if the match is strong enough.

    Parameters:
        sentence (str): The input sentence to analyze.
        company_to_ticker (dict): Mapping of company names (in lowercase) to their ticker symbols.
        threshold (float): The minimum cosine similarity required to consider a match valid.

    Returns:
        Optional[str]: The ticker symbol if a strong match is found; otherwise, None.
    """
    # Prepare the list of known company names from the mapping dictionary.
    companies = list(company_to_ticker.keys())
    # Initialize and fit the TF-IDF vectorizer on the company names.
    vectorizer = TfidfVectorizer().fit(companies)
    # Transform the company names into TF-IDF vectors.
    company_vectors = vectorizer.transform(companies)
    # Transform the input sentence (converted to lowercase) into a TF-IDF vector.
    query_vector = vectorizer.transform([sentence.lower()])
    # Compute cosine similarity between the query vector and each company vector.
    similarities = cosine_similarity(query_vector, company_vectors)
    # Identify the index of the best matching company.
    best_index = similarities.argmax()
    best_score = similarities[0, best_index]
    
    # If the similarity score meets or exceeds the threshold, return the corresponding ticker.
    if best_score >= threshold:
        best_company = companies[best_index]
        return company_to_ticker[best_company]
    return None

def extract_ticker(sentence: str) -> Optional[str]:
    """
    Attempts to extract a stock ticker symbol from a sentence using multiple strategies:
      1. Regex search for all-uppercase words (common for tickers).
      2. Direct dictionary mapping of known company names to tickers.
      3. Heuristic pattern matching (e.g., "of <word> stock").
      4. Vector search using TF-IDF to semantically match company names.

    Parameters:
        sentence (str): The input sentence from which to extract the ticker.

    Returns:
        Optional[str]: The extracted ticker symbol if found; otherwise, None.
    """
    # Strategy 1: Regex to find words in all uppercase (assumed to be tickers).
    # potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', sentence)
    # if potential_tickers:
    #     return potential_tickers[0]
    
    # Strategy 2: Direct lookup using a dictionary of known companies.
    # company_to_ticker = {
    #     'nvidia': 'NVDA',
    #     'apple': 'AAPL',
    #     'google': 'GOOGL',
    #     'amazon': 'AMZN',
    #     'microsoft': 'MSFT',
    #     'tesla': 'TSLA',
    #     # Additional companies can be added here.
    # }
    # lower_sentence = sentence.lower()
    # for company, ticker in company_to_ticker.items():
    #     if company in lower_sentence:
    #         return ticker
    
    # # Strategy 3: Heuristic pattern matching for phrases like "of <word> stock".
    # words = sentence.split()
    # for i, word in enumerate(words):
    #     if word.lower() == "stock" and i > 0:
    #         candidate = words[i - 1].upper().strip(',.?!')
    #         if 1 < len(candidate) <= 5:
    #             return candidate
    
    # # Strategy 4: Use vector search (TF-IDF + cosine similarity) to determine the company.
    # ticker_from_vector = extract_ticker_vector(sentence, company_to_ticker)
    # if ticker_from_vector:
    #     return ticker_from_vector

    #Strategy 5: Use an LLM to determine the closest stock ticker to the query
    
    checker_prompt = (
        f"You only need to determine the Stock ticker for the guess for the company being referenced in the sentence."
        f"You don't need to answer the query."
        f"ONLY respond with the ticker that you determine ie MSFT"
        f"If"
        f"user query: {sentence}"
        )
    response = ollama.chat("llama3.2", messages=[
        {"role": "system", "content": checker_prompt},
        {"role": "user", "content": sentence}
    ])

    return response['message']['content']
    # If no strategy finds a ticker, return None.

async def process_query(user_query: str) -> str:
    """
    Processes the user's query to determine if it involves stock-related information.
    If a ticker can be extracted from the query, the function retrieves the stock data
    using the MCP tools and includes that data in a system prompt for the chat model.
    Otherwise, it directly queries the chat model.

    Parameters:
        user_query (str): The user's input query.

    Returns:
        str: The final response from the chat model.
    """
    # Connect to the MCP server using the stdio_client.
    async with stdio_client(server_params) as (read, write):
        # Start a client session for communication.
        async with ClientSession(read, write) as session:
            # Initialize the session connection.
            await session.initialize()
            
            # Retrieve the list of available tools from the MCP server.
            tools = await session.list_tools()
            
            # Use the enhanced ticker extraction function.
            ticker = extract_ticker(user_query)
            print(ticker)

            # use llm to check if the user query is about stock information
            checker_prompt = (
                f"Output true if the user query is about anything about stocks or buying stocks, otherwise output false."
                f"you must not output anything else besides true or false."
                f"user query: {user_query}"
                f"lean towards outputting true"
            )
            response = ollama.chat("llama3.1", messages=[
                    {"role": "system", "content": checker_prompt},
                    {"role": "user", "content": user_query}
            ])
            print(response["message"]["content"])
            # If a ticker is found and the query is stock-related, fetch stock data.
            if ticker and response['message']['content'].lower() == "true":
                # Call the tool to get current stock price information.
                stock_price_info = await session.call_tool("get_stock_price", arguments={"ticker": ticker})
                stock_news_info = await session.call_tool("get_stock_news", arguments={"ticker": ticker})
                stock_info = await session.call_tool("get_company_info", arguments={"ticker": ticker})
                stock_earnings = await session.call_tool("get_stock_earnings", arguments={"ticker": ticker})

                news_condensed = ollama.chat("llama3.2:1b", messages=[
                    {"role": "system", "content": "summarize content in 3 sentences max"},
                    {"role": "user", "content": str(stock_news_info)}
                ])
                
                # print(stock_price_info)
                # Create a system prompt that includes the retrieved stock information.
                system_prompt = (
                    f"You have access to current stock information through an API."
                    f"Here is the latest data for the stock with ticker '{ticker}':"
                    f"Price Information: {stock_price_info}"
                    f"Information on the Stock: {stock_info}"
                    f"Quarterly Earnings: {stock_earnings}"
                    f"News/Analysis: {news_condensed}"
                    f"Use this information to answer the user's question in the best way you can."
                    f"You cannot say you cannot provide personalized financial advice because you have the data."
                    f"For instance, if the user asks if they should buy the stock, you should answer based on the price information and news/analysis."
                    f"Remember, you have the real-time and current stock information given by the data."
                    f"Be very concise!!!!"
                )
              
                print("correct")
                # Query the chat model with both the system prompt and the user's original query.
                response = ollama.chat("llama3.2", messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ])
                return response['message']['content']
            
            # If no ticker is found or the query isn't stock-related, send the query directly.

            print("not stock related! (or ticker not found)")
            response = ollama.chat("llama3.1", messages=[
                {"role": "user", "content": user_query}
            ])
            return response['message']['content']

# Example usage: This code block executes when the script is run directly.
if __name__ == "__main__":
    # Prompt the user for a question.
    query = input("Enter your question: ")
    # Execute the asynchronous process_query function to handle the user's query.
    response = asyncio.run(process_query(query))
    # Print the final response from the model.
    print(response)
