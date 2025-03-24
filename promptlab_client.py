import asyncio
import argparse
import logging
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promptlab-client")

# Set up environment
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PROMPTLAB_SERVER_SCRIPT = os.environ.get("PROMPTLAB_SERVER_SCRIPT", "promptlab_server.py")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

async def process_query(query: str):
    """Process a query through PromptLab and generate a response."""
    logger.info(f"Processing query: '{query}'")
    
    # Set up server parameters
    server_params = StdioServerParameters(
        command="python",
        args=[PROMPTLAB_SERVER_SCRIPT],
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
    
    try:
        # Connect to the PromptLab server
        async with stdio_client(server_params) as (read, write):
            logger.info("Connected to PromptLab server")
            
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Call the optimize_query tool
                result = await session.call_tool(
                    name="optimize_query", 
                    arguments={"query": query}
                )
                
                # Extract result
                if hasattr(result, 'content') and len(result.content) > 0:
                    result_data = json.loads(result.content[0].text)
                    enhanced_query = result_data.get("enhanced_query", query)
                    initial_enhanced_query = result_data.get("initial_enhanced_query")
                    content_type = result_data.get("content_type", "unknown")
                    validation_issues = result_data.get("validation_issues", [])
                    validation_result = result_data.get("validation_result", "UNKNOWN")
                else:
                    logger.warning("No content returned from server")
                    enhanced_query = query
                    initial_enhanced_query = None
                    content_type = "unknown"
                    validation_issues = ["Server returned no content"]
                    validation_result = "UNKNOWN"
                
                # Log adjustment information
                if validation_result == "NEEDS_ADJUSTMENT" and initial_enhanced_query:
                    logger.info(f"Query required adjustment: {validation_result}")
                    for issue in validation_issues:
                        logger.info(f"Validation issue: {issue}")
                
                # Generate response using the enhanced query
                logger.info(f"Generating response using enhanced query")
                response = await llm.ainvoke([HumanMessage(content=enhanced_query)])
                
                # Return all results with adjustment details
                return {
                    "original_query": query,
                    "content_type": content_type, 
                    "initial_enhanced_query": initial_enhanced_query,
                    "enhanced_query": enhanced_query,
                    "validation_issues": validation_issues,
                    "validation_result": validation_result,
                    "response": response.content
                }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {"error": str(e), "original_query": query}

async def main():
    parser = argparse.ArgumentParser(description="PromptLab: AI Query Enhancement")
    parser.add_argument("query", help="The query to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show validation issues and debug info")
    args = parser.parse_args()
    
    # Process the query
    result = await process_query(args.query)
    
    # Display results
    print("\n=== PromptLab Results ===\n")
    print(f"Original Query: {result['original_query']}")
    
    if "content_type" in result:
        print(f"Content Type: {result['content_type']}")
    
    # Show the initial enhanced query if it differs from the final one
    if "initial_enhanced_query" in result and result.get("initial_enhanced_query") != result.get("enhanced_query"):
        print("\nInitial Enhanced Query:")
        print("-" * 80)
        print(result['initial_enhanced_query'])
        print("-" * 80)
        
        if "validation_issues" in result and result["validation_issues"]:
            print("\nValidation Issues Detected:")
            for issue in result["validation_issues"]:
                print(f"- {issue}")
            print("\nAdjusted Query:")
        else:
            print("\nQuery Adjusted:")
    else:
        print("\nEnhanced Query:")
    
    if "enhanced_query" in result:
        print("-" * 80)
        print(result['enhanced_query'])
        print("-" * 80)
    
    # Show more detailed validation information if verbose mode
    if args.verbose and not "initial_enhanced_query" in result and "validation_issues" in result and result["validation_issues"]:
        print("\nValidation Details:")
        for issue in result["validation_issues"]:
            print(f"- {issue}")
    
    if "response" in result:
        print("\nResponse:")
        print("=" * 80)
        print(result['response'])
        print("=" * 80)
    
    if "error" in result:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())