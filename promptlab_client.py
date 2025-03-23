import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, Any, TypedDict, Literal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promptlab-client")

# Set up your OpenAI API key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Environment variables
PERSONA_SERVER_SCRIPT = os.environ.get("PERSONA_SERVER_SCRIPT", "promptlab_server.py")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    from langgraph.graph import StateGraph, END, START
except ImportError as e:
    logger.error(f"Error importing required package: {e}")
    print(f"\n⚠️  Error importing required package: {e}")
    print("\nPlease install the required packages:")
    print("pip install mcp[cli] langchain-openai langgraph>=0.0.20")
    sys.exit(1)

# Define state schema
class QueryState(TypedDict):
    user_query: str
    content_type: str
    enhanced_query: str
    validation_result: str
    final_query: str
    response: str

# Initialize LLM
llm = ChatOpenAI(temperature=0.1)

# 1. Query Classification Node
async def classify_query(state: QueryState) -> Dict:
    logger.info("Classifying query...")
    
    classification_prompt = f"""
    Based on the user query, determine which type of content the user is asking for:
    1. essay - For academic or informational essays
    2. email - For email composition requests
    3. technical - For technical explanations or tutorials
    4. creative - For creative writing like stories or poems
    
    Only respond with one of these four content types.
    
    User query: {state['user_query']}
    """
    
    response = await llm.ainvoke([HumanMessage(content=classification_prompt)])
    content_type = response.content.strip().lower()
    
    # Normalize content type
    if "essay" in content_type:
        content_type = "essay"
    elif "email" in content_type:
        content_type = "email"
    elif "technical" in content_type:
        content_type = "technical"
    elif "creative" in content_type:
        content_type = "creative"
    else:
        content_type = "essay"  # Default
    
    logger.info(f"Classified as: {content_type}")
    return {"content_type": content_type}

# 2. Query Enhancement Node
async def enhance_query(state: QueryState) -> Dict:
    logger.info(f"Enhancing query as {state['content_type']}...")
    
    # Set up server parameters
    server_params = StdioServerParameters(
        command="python",
        args=[PERSONA_SERVER_SCRIPT],
    )
    
    try:
        # Connect to the MCP server
        async with stdio_client(server_params) as (read, write):
            logger.info("Connected to Persona MCP server")
            
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Extract topic using LLM
                extract_prompt = f"""
                Extract the main topic or subject from this query: "{state['user_query']}"
                Respond with only the topic, without explanation or additional text.
                """
                topic_response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
                topic = topic_response.content.strip()
                logger.info(f"Extracted topic: {topic}")
                
                content_type = state['content_type']
                enhanced_query = ""
                
                try:
                    # Call the appropriate tool based on content type
                    # Using the correct 'call_tool' method
                    if content_type == "essay":
                        result = await session.call_tool(
                            name="essay_prompt", 
                            arguments={"topic": topic}
                        )
                        if hasattr(result, 'content') and len(result.content) > 0:
                            enhanced_query = result.content[0].text
                        logger.info("Successfully called essay_prompt")
                    
                    elif content_type == "email":
                        # Extract recipient type
                        recipient_prompt = f"""
                        From this query: "{state['user_query']}"
                        Who is the email recipient? (e.g., boss, colleague, client, etc.)
                        Respond with only the recipient type, without explanation.
                        """
                        recipient_response = await llm.ainvoke([HumanMessage(content=recipient_prompt)])
                        recipient_type = recipient_response.content.strip()
                        
                        result = await session.call_tool(
                            name="email_prompt", 
                            arguments={"recipient_type": recipient_type, "topic": topic}
                        )
                        if hasattr(result, 'content') and len(result.content) > 0:
                            enhanced_query = result.content[0].text
                        logger.info("Successfully called email_prompt")
                    
                    elif content_type == "technical":
                        # Extract audience level
                        audience_prompt = f"""
                        From this query: "{state['user_query']}"
                        What level of audience is this for? (beginner, intermediate, expert, or general)
                        Respond with only the audience level, without explanation.
                        """
                        audience_response = await llm.ainvoke([HumanMessage(content=audience_prompt)])
                        audience = audience_response.content.strip()
                        
                        result = await session.call_tool(
                            name="technical_prompt", 
                            arguments={"topic": topic, "audience": audience}
                        )
                        if hasattr(result, 'content') and len(result.content) > 0:
                            enhanced_query = result.content[0].text
                        logger.info("Successfully called technical_prompt")
                    
                    elif content_type == "creative":
                        # Extract creative genre
                        genre_prompt = f"""
                        From this query: "{state['user_query']}"
                        What creative genre is requested? (story, poem, script, etc.)
                        Respond with only the genre, without explanation.
                        """
                        genre_response = await llm.ainvoke([HumanMessage(content=genre_prompt)])
                        genre = genre_response.content.strip()
                        
                        result = await session.call_tool(
                            name="creative_prompt", 
                            arguments={"genre": genre, "topic": topic}
                        )
                        if hasattr(result, 'content') and len(result.content) > 0:
                            enhanced_query = result.content[0].text
                        logger.info("Successfully called creative_prompt")
                
                except Exception as e:
                    logger.error(f"Error calling MCP tool: {str(e)}")
                    # Create a fallback if tool call fails
                    if content_type == "essay":
                        enhanced_query = f"Write a well-structured essay about {topic} with an introduction, body paragraphs, and conclusion."
                    elif content_type == "email":
                        enhanced_query = f"Write a professional email about {topic} with appropriate formatting and tone."
                    elif content_type == "technical":
                        enhanced_query = f"Provide a clear technical explanation of {topic} suitable for most audiences."
                    elif content_type == "creative":
                        enhanced_query = f"Write a creative piece about {topic} with vivid descriptions and engaging structure."
                
                if not enhanced_query or enhanced_query.strip() == "":
                    logger.warning("Empty response from MCP tool, using fallback template")
                    enhanced_query = f"Write a detailed {content_type} about {topic} with proper structure and formatting."
                
                logger.info(f"Enhanced query created successfully")
                return {"enhanced_query": enhanced_query.strip()}
    
    except Exception as e:
        logger.error(f"Error in enhance_query: {str(e)}", exc_info=True)
        # Fallback template if MCP server fails completely
        fallback = f"""
        Write a well-structured {state['content_type']} about {state['user_query']} that includes:
        - Proper introduction and conclusion
        - Well-organized body sections
        - Clear progression of ideas
        - Appropriate tone and style
        """
        return {"enhanced_query": fallback.strip()}

# 3. Validation Node
async def validate_query(state: QueryState) -> Dict:
    logger.info("Validating enhanced query...")
    
    validation_prompt = f"""
    I need to validate if an enhanced prompt maintains the original user's intent.
    
    Original user query: "{state['user_query']}"
    
    Enhanced prompt:
    {state['enhanced_query']}
    
    Please analyze and answer these questions:
    1. Does the enhanced prompt maintain the key topic/subject from the original query?
    2. Are there any important elements from the original query that are missing?
    3. Is the content type (essay/email/technical/creative) appropriate for the user's request?
    
    Based on your analysis, respond with ONLY ONE of these verdicts:
    - "VALID" - if the enhanced prompt maintains the original intent and topic
    - "NEEDS_ADJUSTMENT" - if the enhanced prompt misses key elements or changes the topic
    
    Don't include any other text in your response.
    """
    
    response = await llm.ainvoke([HumanMessage(content=validation_prompt)])
    validation_result = response.content.strip().upper()
    
    # Normalize the validation result
    if "VALID" in validation_result:
        result = "VALID"
    else:
        result = "NEEDS_ADJUSTMENT"
    
    logger.info(f"Validation result: {result}")
    return {"validation_result": result}

# 4. Adjustment Node
async def adjust_query(state: QueryState) -> Dict:
    logger.info("Adjusting enhanced query...")
    
    adjustment_prompt = f"""
    I need to adjust an enhanced prompt to better match the user's original request.
    
    Original user query: "{state['user_query']}"
    
    Current enhanced prompt:
    {state['enhanced_query']}
    
    Please create an improved version that:
    1. Maintains all key topics/subjects from the original user query
    2. Keeps the structured format and guidance from the enhanced prompt
    3. Ensures the content type matches what the user wanted
    4. Preserves any specific requirements or constraints mentioned by the user
    
    Provide only the revised enhanced prompt without explanation.
    """
    
    response = await llm.ainvoke([HumanMessage(content=adjustment_prompt)])
    adjusted_query = response.content.strip()
    
    logger.info("Query adjusted successfully")
    return {"final_query": adjusted_query}

# 5. Final Response Generation Node
async def generate_response(state: QueryState) -> Dict:
    logger.info("Generating final response...")
    
    # Use the final query if it exists, otherwise use the enhanced query
    prompt = state.get('final_query', state['enhanced_query'])
    
    # Generate the response
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    logger.info("Response generated successfully")
    return {"response": response.content}

# Decision node to determine next step based on validation
def route_based_on_validation(state: QueryState) -> Literal["adjust", "generate"]:
    if state["validation_result"] == "NEEDS_ADJUSTMENT":
        return "adjust"
    else:
        return "generate"

# Build the LangGraph workflow
async def build_and_run_workflow(user_query: str) -> Dict:
    # Create the graph
    workflow = StateGraph(QueryState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("enhance", enhance_query)
    workflow.add_node("validate", validate_query)
    workflow.add_node("adjust", adjust_query)
    workflow.add_node("generate", generate_response)
    
    # Define edges
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "enhance")
    workflow.add_edge("enhance", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_based_on_validation,
        {
            "adjust": "adjust",
            "generate": "generate"
        }
    )
    workflow.add_edge("adjust", "generate")
    workflow.add_edge("generate", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Initialize state with user query
    initial_state = {"user_query": user_query}
    
    # Run the workflow
    result = await app.ainvoke(initial_state)
    return result

# Main function
async def main():
    parser = argparse.ArgumentParser(description="Process a query through the Persona workflow")
    parser.add_argument("query", help="The user query to process")
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        print("⚠️ Warning: OPENAI_API_KEY not set. Please set it to generate responses.")
        print("export OPENAI_API_KEY=your_key_here")
    
    try:
        print(f"\nProcessing query: '{args.query}'")
        print("-" * 80)
        
        result = await build_and_run_workflow(args.query)
        
        # Display results
        print("\n=== Query Processing Results ===\n")
        print(f"Original Query: {result['user_query']}")
        print(f"Classified as: {result['content_type']}")
        print("\nEnhanced Prompt:")
        print("-" * 80)
        print(result['enhanced_query'])
        print("-" * 80)
        
        print(f"\nValidation Result: {result['validation_result']}")
        
        if result.get('final_query'):
            print("\nAdjusted Final Prompt:")
            print("-" * 80)
            print(result['final_query'])
            print("-" * 80)
        
        print("\nGenerated Response:")
        print("=" * 80)
        print(result['response'])
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())