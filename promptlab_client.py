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

async def list_available_prompts():
    """List all available prompts from MLflow Prompt Registry."""
    logger.info("Listing available prompts")
    
    # Set up server parameters
    server_params = StdioServerParameters(
        command="python",
        args=[PROMPTLAB_SERVER_SCRIPT],
    )
    
    try:
        # Connect to the PromptLab server
        async with stdio_client(server_params) as (read, write):
            logger.info("Connected to PromptLab server")
            
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Call the list_prompts tool
                result = await session.call_tool(
                    name="list_prompts", 
                    arguments={}
                )
                
                # Extract result
                if hasattr(result, 'content') and len(result.content) > 0:
                    result_data = json.loads(result.content[0].text)
                    prompts = result_data.get("prompts", [])
                    
                    print("\n=== Available Prompts ===\n")
                    for prompt in prompts:
                        print(f"Name: {prompt['name']}")
                        print(f"Type: {prompt['type']}")
                        print(f"Version: {prompt.get('version', 'unknown')}")
                        if prompt.get('variables'):
                            print(f"Variables: {', '.join(prompt['variables'])}")
                        if prompt.get('tags'):
                            print(f"Tags: {prompt['tags']}")
                        print()
                    
                    if not prompts:
                        print("No prompts found in the registry. Initialize prompts using 'register_prompts.py register-samples'.")
                else:
                    logger.warning("No content returned from server")
                    print("Failed to retrieve prompts. Check server logs for details.")
    
    except Exception as e:
        logger.error(f"Error listing prompts: {e}", exc_info=True)
        print(f"Error: {str(e)}")

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
                    content_type = result_data.get("content_type")
                    validation_issues = result_data.get("validation_issues", [])
                    validation_result = result_data.get("validation_result", "UNKNOWN")
                    prompt_match = result_data.get("prompt_match", {})
                    was_enhanced = result_data.get("enhanced", False)
                else:
                    logger.warning("No content returned from server")
                    enhanced_query = query
                    initial_enhanced_query = None
                    content_type = None
                    validation_issues = ["Server returned no content"]
                    validation_result = "UNKNOWN"
                    prompt_match = {"status": "unknown"}
                    was_enhanced = False
                
                # Log adjustment information
                if validation_result == "NEEDS_ADJUSTMENT" and initial_enhanced_query:
                    logger.info(f"Query required adjustment: {validation_result}")
                    for issue in validation_issues:
                        logger.info(f"Validation issue: {issue}")
                
                # Generate response using the enhanced query
                logger.info(f"Generating response using {'enhanced' if was_enhanced else 'original'} query")
                response = await llm.ainvoke([HumanMessage(content=enhanced_query)])
                
                # Return all results with adjustment details
                return {
                    "original_query": query,
                    "content_type": content_type, 
                    "enhanced": was_enhanced,
                    "prompt_match": prompt_match,
                    "initial_enhanced_query": initial_enhanced_query if was_enhanced else None,
                    "enhanced_query": enhanced_query,
                    "validation_issues": validation_issues,
                    "validation_result": validation_result,
                    "response": response.content
                }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {"error": str(e), "original_query": query, "enhanced": False}

async def main():
    parser = argparse.ArgumentParser(description="PromptLab: AI Query Enhancement with MLflow")
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show validation issues and debug info")
    parser.add_argument("--list", action="store_true", help="List available prompts from MLflow")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Set temperature from args
    global MODEL_NAME
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    
    if args.list:
        await list_available_prompts()
    elif args.query:
        # Process the query
        result = await process_query(args.query)
        
        # Display results
        print("\n=== PromptLab Results ===\n")
        print(f"Original Query: {result['original_query']}")
        
        # Show prompt matching information
        prompt_match = result.get("prompt_match", {})
        match_status = prompt_match.get("status", "unknown")
        
        if match_status == "matched":
            print(f"\nMatched to: {prompt_match.get('prompt_name')}")
            print(f"Confidence: {prompt_match.get('confidence')}%")
            if args.verbose and prompt_match.get('reasoning'):
                print(f"Reasoning: {prompt_match.get('reasoning')}")
        elif match_status == "no_match":
            print("\nNo matching prompt template found.")
            if args.verbose and prompt_match.get('reasoning'):
                print(f"Reason: {prompt_match.get('reasoning')}")
        elif match_status == "no_prompts_available":
            print("\nNo prompts available in MLflow registry.")
        
        if result.get("content_type"):
            print(f"Content Type: {result['content_type']}")
        
        # Show the enhanced query details if enhancement was performed
        if result.get("enhanced", False):
            # Show the initial enhanced query if it differs from the final one
            if result.get("initial_enhanced_query") and result.get("initial_enhanced_query") != result.get("enhanced_query"):
                print("\nInitial Enhanced Query:")
                print("-" * 80)
                print(result['initial_enhanced_query'])
                print("-" * 80)
                
                if result.get("validation_issues"):
                    print("\nValidation Issues Detected:")
                    for issue in result["validation_issues"]:
                        print(f"- {issue}")
                    print("\nAdjusted Query:")
                else:
                    print("\nQuery Adjusted:")
            else:
                print("\nEnhanced Query:")
            
            if result.get("enhanced_query"):
                print("-" * 80)
                print(result['enhanced_query'])
                print("-" * 80)
            
            # Show more detailed validation information if verbose mode
            if args.verbose and not result.get("initial_enhanced_query") and result.get("validation_issues"):
                print("\nValidation Details:")
                for issue in result["validation_issues"]:
                    print(f"- {issue}")
        else:
            print("\nUsing original query (no enhancement applied)")
        
        if result.get("response"):
            print("\nResponse:")
            print("=" * 80)
            print(result['response'])
            print("=" * 80)
        
        if result.get("error"):
            print(f"\nError: {result['error']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())