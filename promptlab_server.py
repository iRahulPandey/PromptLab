from mcp.server.fastmcp import FastMCP
import logging
import yaml
import os
import asyncio
import sys
from typing import Dict, Any, TypedDict

# Import LangGraph components
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promptlab-server")

# Set up your OpenAI API key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# File paths
TEMPLATES_FILE = os.environ.get("TEMPLATES_FILE", "prompt_templates.yaml")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define state schema for type safety
class QueryState(TypedDict, total=False):
    user_query: str
    content_type: str
    enhanced_query: str
    validation_result: str
    final_query: str
    validation_issues: list

# Initialize MCP server
mcp = FastMCP(
    name="promptlab",
    instructions="AI query enhancement service that transforms basic queries into optimized prompts."
)

# Load templates
def load_templates(file_path):
    """Load templates from YAML file."""
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        logger.info(f"Successfully loaded templates from {file_path}")
        return data.get('templates', {})
    except Exception as e:
        logger.error(f"Error loading templates from {file_path}: {e}")
        return {}

templates = load_templates(TEMPLATES_FILE)

# Initialize LLM with proper error handling
if OPENAI_API_KEY:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.1)
    logger.info("ChatOpenAI initialized with API key")
else:
    llm = None
    logger.warning("No OpenAI API key found - using rule-based processing only")

# Define workflow nodes
async def classify_query(state: QueryState) -> QueryState:
    """Classify the content type of the query."""
    logger.info(f"Classifying query: {state['user_query']}")
    
    query = state['user_query']
    
    if llm:
        # Use LLM for more accurate classification
        classification_prompt = f"""
        Based on the user query, determine which type of content the user is asking for:
        1. essay - For academic or informational essays
        2. email - For email composition requests
        3. technical - For technical explanations or tutorials
        4. creative - For creative writing like stories or poems
        
        Only respond with one of these four content types.
        
        User query: {query}
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
    else:
        # Simple classification logic as fallback
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["essay", "write about", "discuss"]):
            content_type = "essay"
        elif any(kw in query_lower for kw in ["email", "message to", "write to"]):
            content_type = "email"
        elif any(kw in query_lower for kw in ["explain", "how does", "technical"]):
            content_type = "technical"
        elif any(kw in query_lower for kw in ["story", "creative", "poem", "fiction"]):
            content_type = "creative"
        else:
            # Default to essay if uncertain
            content_type = "essay"
    
    logger.info(f"Classified as: {content_type}")
    # Return a new dictionary with the original state AND the new content_type
    return {"user_query": query, "content_type": content_type}

async def enhance_query(state: QueryState) -> QueryState:
    """Apply the appropriate template to enhance the query."""
    logger.info(f"Enhancing query as {state['content_type']}...")
    
    content_type = state['content_type']
    query = state['user_query']
    current_llm = state.get('llm', llm)  # Use provided LLM or fall back to default
    
    # Extract key parameters using LLM if available
    if current_llm and content_type == "creative":
        # Extract topic and genre
        extract_prompt = f"""
        Extract from this query: "{query}"
        1. The main topic or subject
        2. The creative genre (story, poem, script, etc.)

        Format your response as a JSON object with two fields:
        topic: The main topic or subject
        genre: The creative genre
        
        Example:
        {{
            "topic": "magical forest",
            "genre": "story"
        }}
        """
        
        try:
            response = await current_llm.ainvoke([HumanMessage(content=extract_prompt)])
            import json
            params = json.loads(response.content)
            topic = params.get("topic", "")
            genre = params.get("genre", "story")
        except Exception as e:
            logger.warning(f"Error extracting parameters with LLM: {e}")
            # Fallback to simple extraction
            topic = query.replace("write", "").replace("about", "").strip()
            genre = "story"
            for possible_genre in ["story", "poem", "script", "novel"]:
                if possible_genre in query.lower():
                    genre = possible_genre
                    break
    elif current_llm and content_type == "email":
        # Extract more detailed parameters for email
        extract_prompt = f"""
        Extract from this query: "{query}"
        1. The recipient type (boss, colleague, friend, etc.)
        2. The main topic of the email

        Format your response as a JSON object with two fields:
        recipient_type: The type of recipient
        topic: The main topic of the email
        
        Example:
        {{
            "recipient_type": "boss",
            "topic": "vacation request"
        }}
        """
        
        try:
            response = await current_llm.ainvoke([HumanMessage(content=extract_prompt)])
            import json
            params = json.loads(response.content)
            topic = params.get("topic", "")
            recipient_type = params.get("recipient_type", "")
        except Exception as e:
            logger.warning(f"Error extracting parameters with LLM: {e}")
            # Fallback to simple extraction
            topic = query.replace("write", "").replace("email", "").replace("to", "").replace("about", "").strip()
            recipient_type = "recipient"
            # Try to find recipient in the query
            if "to my" in query.lower():
                recipient_parts = query.lower().split("to my")
                if len(recipient_parts) > 1:
                    recipient_type = recipient_parts[1].split()[0]
            elif "to" in query.lower():
                recipient_parts = query.lower().split("to")
                if len(recipient_parts) > 1:
                    possible_recipient = recipient_parts[1].split()[0]
                    if possible_recipient not in ["the", "a", "an"]:
                        recipient_type = possible_recipient
    else:
        # Simple parameter extraction for other types or when LLM not available
        topic = query.replace("write", "").replace("about", "").strip()
        genre = "story" if content_type == "creative" else ""
        recipient_type = "recipient" if content_type == "email" else ""
    
    # Select template based on content type
    template_name = f"{content_type}_prompt"
    enhanced_query = ""
    
    if template_name in templates:
        template = templates[template_name]
        
        # Prepare parameters based on content type
        params = {"topic": topic}
        
        if content_type == "email":
            params["recipient_type"] = recipient_type
            
        elif content_type == "technical":
            params["audience"] = "general"
            
        elif content_type == "creative":
            params["genre"] = genre
        
        # Apply transformations if present in the template
        if 'transformations' in template:
            for transform in template['transformations']:
                transform_name = transform['name']
                transform_expr = transform['value']
                
                try:
                    # Handle conditional expressions (if/else)
                    if "if" in transform_expr and "else" in transform_expr:
                        # Parse the conditional expression
                        condition_parts = transform_expr.split(" if ")
                        if len(condition_parts) == 2:
                            true_value = condition_parts[0].strip().strip('"\'')
                            condition_else_parts = condition_parts[1].split(" else ")
                            if len(condition_else_parts) == 2:
                                condition = condition_else_parts[0].strip()
                                false_value = condition_else_parts[1].strip().strip('"\'')
                                
                                # Create a safe evaluation environment
                                eval_locals = {}
                                for param_name, param_value in params.items():
                                    if isinstance(param_value, str):
                                        eval_locals[param_name] = param_value.lower()
                                    else:
                                        eval_locals[param_name] = param_value
                                
                                # Evaluate the condition
                                try:
                                    condition_result = eval(condition, {"__builtins__": {}}, eval_locals)
                                    params[transform_name] = true_value if condition_result else false_value
                                except Exception as e:
                                    logger.error(f"Error evaluating condition '{condition}': {e}")
                                    # Use a default value
                                    params[transform_name] = false_value
                    else:
                        # For simple expressions without conditionals
                        params[transform_name] = transform_expr
                except Exception as e:
                    logger.error(f"Error applying transformation '{transform_name}': {e}")
                    # Set a default value to prevent template formatting errors
                    if transform_name == "formality":
                        params[transform_name] = "formal" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "semi-formal"
                    elif transform_name == "tone":
                        params[transform_name] = "respectful and professional" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "friendly but professional"
                    else:
                        params[transform_name] = "default"
        
        # Format the template - handle any missing parameters
        try:
            logger.info(f"Formatting template with parameters: {params}")
            enhanced_query = template['template'].format(**params)
        except KeyError as e:
            logger.error(f"Missing parameter for template: {e}")
            # Try to add the missing parameter with a default value
            missing_param = str(e).strip("'")
            if missing_param == "formality":
                params["formality"] = "formal" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "semi-formal"
            elif missing_param == "tone":
                params["tone"] = "respectful and professional" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "friendly but professional"
            else:
                params[missing_param] = "appropriate"
            
            try:
                enhanced_query = template['template'].format(**params)
            except Exception as nested_e:
                logger.error(f"Error after adding default parameter: {nested_e}")
                enhanced_query = f"Write a {content_type} about {topic}"
                
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            enhanced_query = f"Write a {content_type} about {topic}"
    else:
        # Fallback for unknown template
        enhanced_query = f"Write a {content_type} about {topic}"
    
    logger.info("Query enhanced successfully")
    # Return a merged dictionary with ALL previous state plus new fields
    return {**state, "enhanced_query": enhanced_query}

async def validate_query(state: QueryState) -> QueryState:
    """Validate that the enhanced query maintains the original intent and is well-formed."""
    logger.info("Validating enhanced query...")
    
    user_query = state['user_query']
    enhanced_query = state['enhanced_query']
    validation_issues = []
    
    # Check for repeated phrases or words (sign of a formatting issue)
    parts = user_query.lower().split()
    for part in parts:
        if len(part) > 4:  # Only check substantial words
            count = enhanced_query.lower().count(part)
            if count > 1:
                validation_issues.append(f"Repeated word: '{part}'")
    
    # Check for direct inclusion of the user query
    if user_query.lower() in enhanced_query.lower():
        validation_issues.append("Raw user query inserted into template")
    
    # Check if major words from the original query are present in the enhanced version
    major_words = [word for word in user_query.lower().split() 
                if len(word) > 4 and word not in ["write", "about", "create", "make"]]
    
    missing_words = [word for word in major_words 
                   if word not in enhanced_query.lower()]
    
    if missing_words:
        validation_issues.append(f"Missing key words: {', '.join(missing_words)}")
    
    # Always validate using LLM if available for more sophisticated checks
    if llm:
        validation_prompt = f"""
        I need to validate if an enhanced prompt maintains the original user's intent and is well-formed.
        
        Original user query: "{user_query}"
        
        Enhanced prompt:
        {enhanced_query}
        
        Please analyze and identify any issues:
        1. Does the enhanced prompt maintain the key topic/subject from the original query?
        2. Are there any important elements from the original query that are missing?
        3. Is the enhanced prompt well-formed (no repeated words, no grammatical errors)?
        4. Does the enhanced prompt avoid directly inserting the raw user query?
        
        Return a JSON object with:
        - "valid": boolean (true if no issues, false otherwise)
        - "issues": array of string descriptions of any problems found (empty if valid)
        """
        
        try:
            response = await llm.ainvoke([HumanMessage(content=validation_prompt)])
            import json
            validation_result = json.loads(response.content)
            
            if not validation_result.get("valid", False):
                validation_issues.extend(validation_result.get("issues", []))
        except Exception as e:
            logger.warning(f"Error validating with LLM: {e}")
            # Continue with rule-based validation results
    
    # Determine final validation result
    final_validation = "NEEDS_ADJUSTMENT" if validation_issues else "VALID"
    
    logger.info(f"Validation result: {final_validation}")
    # Return merged state
    return {**state, "validation_result": final_validation, "validation_issues": validation_issues}

async def adjust_query(state: QueryState) -> QueryState:
    """Adjust the enhanced query to address validation issues."""
    logger.info("Adjusting enhanced query...")
    
    enhanced_query = state['enhanced_query']
    user_query = state['user_query']
    validation_issues = state.get('validation_issues', [])
    
    # LLM-based adjustment for more sophisticated fixes
    if llm:
        adjustment_prompt = f"""
        I need to adjust an enhanced prompt to better match the user's original request and fix identified issues.
        
        Original user query: "{user_query}"
        
        Current enhanced prompt:
        {enhanced_query}
        
        Issues that need to be fixed:
        {', '.join(validation_issues)}
        
        Please create an improved version that:
        1. Maintains all key topics/subjects from the original user query
        2. Keeps the structured format and guidance of a prompt template
        3. Ensures the content type matches what the user wanted
        4. Fixes all the identified issues
        5. Does NOT include the raw user query directly in the text
        
        Provide only the revised enhanced prompt without explanation or metadata.
        """
        
        try:
            response = await llm.ainvoke([HumanMessage(content=adjustment_prompt)])
            adjusted_query = response.content.strip()
        except Exception as e:
            logger.warning(f"Error adjusting with LLM: {e}")
            # Fall back to simple adjustments
            adjusted_query = enhanced_query
            
            # Simple rule-based adjustments as fallback
            for issue in validation_issues:
                if "Repeated word" in issue:
                    # Try to fix repetitions
                    word = issue.split("'")[1]
                    parts = adjusted_query.split(word)
                    if len(parts) > 2:  # More than one occurrence
                        adjusted_query = parts[0] + word + "".join(parts[2:])
                
                if "Raw user query inserted" in issue:
                    # Try to remove the raw query
                    adjusted_query = adjusted_query.replace(user_query, "")
                    
                if "Missing key words" in issue:
                    # Try to add missing words
                    missing = issue.split(": ")[1]
                    adjusted_query = f"{adjusted_query}\nPlease include these key elements: {missing}"
    else:
        # Simple rule-based adjustments
        adjusted_query = enhanced_query
        
        # Fix common issues without LLM
        # 1. Handle repetitions
        words = user_query.lower().split()
        for word in words:
            if len(word) > 4 and enhanced_query.lower().count(word) > 1:
                parts = enhanced_query.split(word)
                if len(parts) > 2:
                    adjusted_query = parts[0] + word + "".join(parts[2:])
        
        # 2. Handle raw query insertion
        if user_query in adjusted_query:
            topic = user_query.replace("write", "").replace("about", "").strip()
            adjusted_query = adjusted_query.replace(user_query, topic)
        
        # 3. Add missing words
        missing_words = [word for word in words if len(word) > 4 and word not in adjusted_query.lower()]
        if missing_words:
            missing_terms = ", ".join(missing_words)
            adjusted_query = f"{adjusted_query}\nPlease include these key elements: {missing_terms}"
    
    logger.info("Query adjusted successfully")
    # Return merged state with final query
    return {**state, "final_query": adjusted_query}

# Conditional routing function
def route_based_on_validation(state: QueryState) -> str:
    if state["validation_result"] == "NEEDS_ADJUSTMENT":
        return "adjust"
    else:
        return "generate"

# Final generation node
async def generate_final_result(state: QueryState) -> QueryState:
    """Generate the final result with the optimized query."""
    # Select the best query to use - final_query if available, otherwise enhanced_query
    final_query = state.get("final_query") or state["enhanced_query"]
    
    # Return the state with a cleaned-up final result
    return {**state, "final_query": final_query}

# Define the complete workflow
def create_workflow():
    """Create and return the LangGraph workflow."""
    # Create the graph with type hints
    workflow = StateGraph(QueryState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("enhance", enhance_query)
    workflow.add_node("validate", validate_query)
    workflow.add_node("adjust", adjust_query)
    workflow.add_node("generate", generate_final_result)
    
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
    return workflow.compile()

# Initialize the workflow
workflow_app = create_workflow()

@mcp.tool()
async def optimize_query(query: str) -> Dict[str, Any]:
    """
    Optimize a user query by applying the appropriate template using LangGraph workflow.
    
    Args:
        query: The original user query
        
    Returns:
        Enhanced query and metadata
    """
    logger.info(f"Processing query through workflow: {query}")
    
    try:
        # Initial state
        initial_state = {"user_query": query}
        
        # Run the workflow
        result = await workflow_app.ainvoke(initial_state)
        
        # Prepare response with more detailed information about the process
        response = {
            "original_query": query,
            "content_type": result.get("content_type", "unknown"),
            "initial_enhanced_query": result.get("enhanced_query", query) if result.get("validation_result") == "NEEDS_ADJUSTMENT" else None,
            "enhanced_query": result.get("final_query") or result.get("enhanced_query", query),
            "validation_result": result.get("validation_result", "UNKNOWN"),
            "validation_issues": result.get("validation_issues", [])
        }
        
        logger.info(f"Successfully processed query")
        return response
    except Exception as e:
        logger.error(f"Error in workflow: {e}", exc_info=True)
        return {
            "original_query": query,
            "enhanced_query": query,  # Return original as fallback
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info(f"Starting PromptLab server with {len(templates)} templates")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)