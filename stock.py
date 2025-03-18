
import httpx
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import json

# Create an MCP server
mcp = FastMCP("Yahoo Finance Assistant")

@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price and basic information for a ticker symbol.
    
    Parameters:
    - ticker: The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    
    Returns:
    - Basic stock information including current price and daily change
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        previous_close = info.get('previousClose', 'N/A')
        company_name = info.get('shortName', info.get('longName', ticker))
        
        # Calculate percent change
        if current_price != 'N/A' and previous_close != 'N/A':
            change = current_price - previous_close
            percent_change = (change / previous_close) * 100
            change_str = f"{change:.2f} ({percent_change:.2f}%)"
        else:
            change_str = "N/A"
        
        return (f"Stock: {company_name} ({ticker})\n"
               f"Current Price: ${current_price}\n"
               f"Change: {change_str}\n"
               f"Previous Close: ${previous_close}")
    except Exception as e:
        return f"Error retrieving stock data for {ticker}: {str(e)}"

@mcp.tool()
def get_financial_statements(ticker: str, statement_type: str = "income") -> str:
    """
    Get financial statements for a ticker symbol.
    
    Parameters:
    - ticker: The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    - statement_type: Type of financial statement ("income", "balance", "cash")
    
    Returns:
    - Financial statement data in a formatted string
    """
    try:
        stock = yf.Ticker(ticker)
        
        if statement_type.lower() == "income":
            statement = stock.financials
            statement_name = "Income Statement"
        elif statement_type.lower() == "balance":
            statement = stock.balance_sheet
            statement_name = "Balance Sheet"
        elif statement_type.lower() == "cash":
            statement = stock.cashflow
            statement_name = "Cash Flow Statement"
        else:
            return f"Invalid statement type: {statement_type}. Choose from: income, balance, cash"
        
        # Format the statement for readability
        if statement.empty:
            return f"No {statement_name} data available for {ticker}"
        
        # Convert to string representation with formatting
        result = f"{statement_name} for {ticker}:\n\n"
        
        # Format the DataFrame as a string table
        statement_str = statement.head(10).to_string()
        
        return result + statement_str
    except Exception as e:
        return f"Error retrieving financial statement for {ticker}: {str(e)}"

@mcp.tool()
def get_company_info(ticker: str) -> str:
    """
    Get detailed company information for a ticker symbol.
    
    Parameters:
    - ticker: The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    
    Returns:
    - Detailed company information including sector, industry, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key company information
        company_name = info.get('shortName', info.get('longName', ticker))
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = f"${market_cap/1000000000:.2f} billion"
        
        pe_ratio = info.get('trailingPE', info.get('forwardPE', 'N/A'))
        if pe_ratio != 'N/A':
            pe_ratio = f"{pe_ratio:.2f}"
        
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield = f"{dividend_yield*100:.2f}%"
        
        return (f"Company: {company_name} ({ticker})\n"
               f"Sector: {sector}\n"
               f"Industry: {industry}\n"
               f"Market Cap: {market_cap}\n"
               f"P/E Ratio: {pe_ratio}\n"
               f"Dividend Yield: {dividend_yield}")
    except Exception as e:
        return f"Error retrieving company information for {ticker}: {str(e)}"
    
@mcp.tool()
def get_stock_earnings(ticker: str) -> str:
    """
    Get the quarterly earnings of a stock.
    """
    try:
        stock = yf.Ticker(ticker)
        return (
            f"Earnings for {ticker}: {stock.earnings}"
        )
    except Exception as e:
        return f"Error retrieving company earnings for {ticker}: {str(e)}"

@mcp.tool()
def get_stock_news(ticker: str) -> str:
    """
    Get news about a stock.
    
    Parameters:
    - ticker: The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    
    Returns:
    - Detailed company information including sector, industry, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        
        return (
            f"News about {ticker}: {stock.news}"
        )
    except Exception as e:
        return f"Error retrieving company news for {ticker}: {str(e)}"

if __name__ == "__main__":
    mcp.run()
