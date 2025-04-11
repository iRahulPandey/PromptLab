from mcp.server.fastmcp import FastMCP
import logging
import os
import asyncio
import sys
import json
from typing import Dict, Any, TypedDict, List, Optional, Tuple

# Import LangGraph components
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import MLflow components for prompt management
import mlflow
from mlflow.entities import Prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promptlab-server")

# Set up environment variables
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')

# Define state schema for type safety
class QueryState(TypedDict, total=False):
    user_query: str
    content_type: Optional[str]
    prompt_match: Dict[str, Any]  # Store matched prompt info
    enhanced_query: str
    validation_result: str
    final_query: str
    validation_issues: list
    available_prompts: Dict[str, Prompt]  # Store available prompts for access in the workflow
    should_skip_enhance: bool  # Flag to indicate if enhancement should be skipped
    parameters: Dict[str, Any]  # Extracted parameters for template

# Initialize MCP server
mcp = FastMCP(
    name="promptlab",
    instructions="AI query enhancement service that transforms basic queries into optimized prompts using MLflow Prompt Registry."
)

# Initialize LLM with proper error handling
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.1)
    logger.info("ChatOpenAI initialized with API key")
else:
    llm = None
    logger.warning("No OpenAI API key found - using rule-based processing only")

# MLflow connection setup
def setup_mlflow_connection():
    """Setup connection to MLflow server."""
    # Set MLflow tracking URI if not already set
    if not os.environ["MLFLOW_TRACKING_URI"]:
        mlflow_uri = os.path.abspath("./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Set MLflow tracking URI to local directory: {mlflow_uri}")
    else:
        mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Using MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")

# Load prompts from MLflow
def load_all_prompts() -> Dict[str, Prompt]:
    """
    Load all available prompts from MLflow Prompt Registry.
    
    Returns:
        Dictionary mapping prompt names to their corresponding prompt objects
    """
    prompts = {}
    
    # Ensure MLflow connection is set up
    setup_mlflow_connection()
    
    # Try to find all available prompts by checking known naming patterns
    # We'll look for both standard patterns and any custom prompts
    
    # Known prompt naming patterns
    known_prompt_types = [
        "essay", "email", "technical", "creative", "code", 
        "summary", "analysis", "qa", "custom", "social_media", 
        "blog", "report", "letter", "presentation", "review",
        "comparison", "instruction"
    ]
    
    for prompt_type in known_prompt_types:
        prompt_name = f"{prompt_type}_prompt"
        
        # First try with production alias (preferred)
        try:
            prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@production")
            logger.info(f"Loaded prompt '{prompt_name}' with production alias (version {prompt.version})")
            prompts[prompt_name] = prompt
            continue
        except Exception as e:
            logger.debug(f"Could not load prompt '{prompt_name}@production': {e}")
        
        # If production alias fails, try the latest version
        try:
            prompt = mlflow.load_prompt(f"prompts:/{prompt_name}")
            logger.info(f"Loaded latest version of prompt '{prompt_name}' (version {prompt.version})")
            prompts[prompt_name] = prompt
        except Exception as e:
            logger.debug(f"Could not load any version of prompt '{prompt_name}': {e}")
    
    # Add additional custom prompts if they follow a different naming convention
    # This would need to be expanded based on your prompt naming strategy
    
    logger.info(f"Loaded {len(prompts)} prompts from MLflow")
    return prompts

# Define workflow nodes
async def load_prompts(state: QueryState) -> QueryState:
    """Load all available prompts from MLflow as the first step in the workflow."""
    logger.info("Loading prompts from MLflow")
    
    try:
        prompts = load_all_prompts()
        logger.info(f"Successfully loaded {len(prompts)} prompts")
        return {**state, "available_prompts": prompts}
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        # Return empty dict of prompts which will trigger skip-enhance path
        return {**state, "available_prompts": {}}

async def match_prompt(state: QueryState) -> QueryState:
    """
    Use LLM to match the user query to the most appropriate prompt template.
    This replaces the hardcoded content type classification.
    """
    logger.info(f"Matching query to prompt template: {state['user_query']}")
    query = state['user_query']
    available_prompts = state.get('available_prompts', {})
    
    # If no prompts are available, skip enhancement
    if not available_prompts:
        logger.warning("No prompts available, skipping enhancement")
        return {
            **state, 
            "should_skip_enhance": True,
            "prompt_match": {"status": "no_prompts_available"}
        }
    
    # Available prompt names for selection
    prompt_names = list(available_prompts.keys())
    prompt_details = {}
    
    # Get details about each prompt for better matching
    for name, prompt_obj in available_prompts.items():
        # Extract variables from the template
        variables = []
        template = prompt_obj.template
        import re
        for match in re.finditer(r'{{([^{}]+)}}', template):
            var_name = match.group(1).strip()
            variables.append(var_name)
        
        # Get description from metadata or tags if available
        description = ""
        if hasattr(prompt_obj, "tags") and prompt_obj.tags:
            type_tag = prompt_obj.tags.get("type", "")
            task_tag = prompt_obj.tags.get("task", "")
            description = f"{type_tag} {task_tag}".strip()
        
        # Create a simple description from the name if no tags
        if not description:
            # Convert name like "essay_prompt" to "essay"
            description = name.replace("_prompt", "").replace("_", " ")
        
        prompt_details[name] = {
            "variables": variables,
            "description": description
        }
    
    if llm:
        # Use LLM to match the query to the best prompt
        matching_prompt = f"""
        I need to match a user query to the most appropriate prompt template from a list.

        User query: "{query}"
        
        Available prompt templates:
        {json.dumps(prompt_details, indent=2)}
        
        Choose the most appropriate prompt template for this query based on the intent and requirements.
        If none of the templates are appropriate, respond with "none".
        
        Return your answer as a JSON object with these fields:
        - "prompt_name": The name of the best matching prompt (or "none")
        - "confidence": A number between 0-100 representing your confidence in this match
        - "reasoning": A brief explanation for why this template is appropriate
        - "parameters": A dictionary with values for the template variables extracted from the query
        
        Example response for selecting an email template:
        {{
            "prompt_name": "email_prompt",
            "confidence": 85,
            "reasoning": "The user wants to write a professional email to their boss about a vacation request",
            "parameters": {{
                "recipient_type": "boss",
                "topic": "vacation request",
                "formality": "formal",
                "tone": "respectful and professional"
            }}
        }}
        """
        
        try:
            response = await llm.ainvoke([HumanMessage(content=matching_prompt)])
            match_result = json.loads(response.content)
            
            # Check if a valid prompt was matched
            if match_result.get("prompt_name", "none") != "none" and match_result.get("prompt_name") in available_prompts:
                prompt_name = match_result["prompt_name"]
                logger.info(f"Matched query to '{prompt_name}' with {match_result.get('confidence')}% confidence")
                
                # Get the prompt object
                matched_prompt = available_prompts[prompt_name]
                
                # Extract content type from prompt name (e.g., "essay_prompt" -> "essay")
                content_type = prompt_name.replace("_prompt", "")
                
                # Return state with matched prompt information
                return {
                    **state, 
                    "content_type": content_type,
                    "prompt_match": {
                        "status": "matched",
                        "prompt_name": prompt_name,
                        "confidence": match_result.get("confidence", 0),
                        "reasoning": match_result.get("reasoning", ""),
                    },
                    "parameters": match_result.get("parameters", {}),
                    "should_skip_enhance": False
                }
            else:
                # No appropriate prompt template found
                logger.info("No appropriate prompt template found for query")
                return {
                    **state, 
                    "content_type": None,
                    "prompt_match": {
                        "status": "no_match",
                        "reasoning": match_result.get("reasoning", "No matching template found")
                    },
                    "should_skip_enhance": True
                }
        except Exception as e:
            logger.error(f"Error matching prompt with LLM: {e}")
            # Fall back to simple matching or no enhancement
    
    # Simple fallback matching if LLM fails or isn't available
    # Use basic keyword matching to find an appropriate template
    keyword_matches = {}
    query_lower = query.lower()
    
    keyword_map = {
        "essay": ["essay", "write about", "discuss", "research", "analyze", "academic"],
        "email": ["email", "message", "write to", "contact", "reach out"],
        "technical": ["explain", "how does", "technical", "guide", "tutorial", "concept"],
        "creative": ["story", "creative", "poem", "fiction", "narrative", "imaginative"],
        "code": ["code", "program", "script", "function", "algorithm", "programming", "implement"],
        "summary": ["summarize", "summary", "brief", "condense", "overview", "recap"],
        "analysis": ["analyze", "analysis", "critique", "evaluate", "assess", "examine"],
        "qa": ["question", "answer", "qa", "respond to", "reply to", "doubt"],
        "custom": ["custom", "specialized", "specific", "tailored", "personalized"],
        "social_media": ["post", "tweet", "social media", "instagram", "facebook", "linkedin"],
        "blog": ["blog", "article", "post about", "write blog", "blog post"],
        "report": ["report", "business report", "analysis report", "status", "findings"],
        "letter": ["letter", "formal letter", "cover letter", "recommendation letter"],
        "presentation": ["presentation", "slides", "slideshow", "deck", "talk"],
        "review": ["review", "evaluate", "critique", "assess", "feedback", "opinion"],
        "comparison": ["compare", "comparison", "versus", "vs", "differences", "similarities"],
        "instruction": ["instructions", "how to", "steps", "guide", "tutorial", "directions"]
    }
    
    for prompt_type, keywords in keyword_map.items():
        prompt_name = f"{prompt_type}_prompt"
        if prompt_name in available_prompts:
            for keyword in keywords:
                if keyword in query_lower:
                    keyword_matches[prompt_name] = keyword_matches.get(prompt_name, 0) + 1
    
    # Find the prompt with the most keyword matches
    if keyword_matches:
        best_match = max(keyword_matches.items(), key=lambda x: x[1])
        prompt_name = best_match[0]
        content_type = prompt_name.replace("_prompt", "")
        
        logger.info(f"Matched query to '{prompt_name}' using keyword matching")
        
        # Extract basic parameters
        parameters = {"topic": query.replace("write", "").replace("about", "").strip()}
        
        # Add specific parameters based on prompt type
        if "email_prompt" == prompt_name:
            recipient_type = "recipient"
            if "to my" in query_lower:
                recipient_parts = query_lower.split("to my")
                if len(recipient_parts) > 1:
                    recipient_type = recipient_parts[1].split()[0]
            parameters["recipient_type"] = recipient_type
            parameters["formality"] = "formal" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "semi-formal"
            parameters["tone"] = "respectful" if recipient_type in ['boss', 'supervisor', 'manager', 'client'] else "friendly"
        
        elif "creative_prompt" == prompt_name:
            genre = "story"
            for possible_genre in ["story", "poem", "script", "novel"]:
                if possible_genre in query_lower:
                    genre = possible_genre
                    break
            parameters["genre"] = genre
        
        elif "technical_prompt" == prompt_name:
            parameters["audience"] = "general"
        
        return {
            **state, 
            "content_type": content_type,
            "prompt_match": {
                "status": "matched",
                "prompt_name": prompt_name,
                "confidence": 70,  # Medium confidence for keyword matching
                "reasoning": f"Matched based on keywords in query",
            },
            "parameters": parameters,
            "should_skip_enhance": False
        }
    
    # If no match found, skip enhancement
    logger.info("No matching prompt found, skipping enhancement")
    return {
        **state, 
        "content_type": None,
        "prompt_match": {"status": "no_match"},
        "should_skip_enhance": True
    }

async def enhance_query(state: QueryState) -> QueryState:
    """Apply the matched prompt template to enhance the query."""
    logger.info(f"Enhancing query...")
    
    # Check if we should skip enhancement
    if state.get("should_skip_enhance", False):
        logger.info("Skipping enhancement as requested")
        return {**state, "enhanced_query": state["user_query"]}
    
    prompt_match = state.get("prompt_match", {})
    available_prompts = state.get("available_prompts", {})
    parameters = state.get("parameters", {})
    
    # Get the matched prompt
    prompt_name = prompt_match.get("prompt_name")
    if not prompt_name or prompt_name not in available_prompts:
        logger.warning(f"Prompt '{prompt_name}' not found in available prompts, skipping enhancement")
        return {**state, "enhanced_query": state["user_query"]}
    
    prompt = available_prompts[prompt_name]
    
    try:
        # Format the prompt with extracted parameters
        logger.info(f"Formatting prompt '{prompt_name}' with parameters: {parameters}")
        enhanced_query = prompt.format(**parameters)
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        # Try to fill in missing parameters if possible
        try:
            # Extract required variables from template
            import re
            required_vars = []
            for match in re.finditer(r'{{([^{}]+)}}', prompt.template):
                var_name = match.group(1).strip()
                required_vars.append(var_name)
            
            # Fill in missing parameters with defaults
            for var in required_vars:
                if var not in parameters:
                    # Default values based on common parameter names
                    if var == "topic":
                        parameters[var] = state["user_query"].replace("write", "").replace("about", "").strip()
                    elif var == "audience" or var == "recipient_type":
                        parameters[var] = "general"
                    elif var == "formality":
                        parameters[var] = "formal"
                    elif var == "tone":
                        parameters[var] = "professional"
                    elif var == "genre":
                        parameters[var] = "story"
                    else:
                        parameters[var] = "appropriate"
            
            enhanced_query = prompt.format(**parameters)
        except Exception as nested_e:
            logger.error(f"Error filling in missing parameters: {nested_e}")
            # Fall back to original query
            enhanced_query = state["user_query"]
    
    logger.info("Query enhanced successfully")
    # Return a merged dictionary with ALL previous state plus new fields
    return {**state, "enhanced_query": enhanced_query}

async def validate_query(state: QueryState) -> QueryState:
    """Validate that the enhanced query maintains the original intent and is well-formed."""
    logger.info("Validating enhanced query...")
    
    # If enhancement was skipped, skip validation as well
    if state.get("should_skip_enhance", False):
        logger.info("Enhancement was skipped, skipping validation as well")
        return {**state, "validation_result": "VALID", "validation_issues": []}
    
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
    
    # If enhancement was skipped, skip adjustment as well
    if state.get("should_skip_enhance", False):
        logger.info("Enhancement was skipped, skipping adjustment as well")
        return {**state, "final_query": state["user_query"]}
    
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

# Conditional routing functions
def route_after_match(state: QueryState) -> str:
    """Route based on whether enhancement should be skipped."""
    if state.get("should_skip_enhance", False):
        return "generate"
    else:
        return "enhance"

def route_based_on_validation(state: QueryState) -> str:
    """Route based on validation result."""
    if state.get("should_skip_enhance", False):
        return "generate"
    elif state["validation_result"] == "NEEDS_ADJUSTMENT":
        return "adjust"
    else:
        return "generate"

# Final generation node
async def generate_final_result(state: QueryState) -> QueryState:
    """Generate the final result with the optimized query."""
    # Select the best query to use - final_query if available, otherwise enhanced_query, or original if enhancement was skipped
    if state.get("should_skip_enhance", False):
        logger.info("Using original query as enhancement was skipped")
        final_query = state["user_query"]
    else:
        final_query = state.get("final_query") or state.get("enhanced_query") or state["user_query"]
    
    # Return the state with a cleaned-up final result
    return {**state, "final_query": final_query}

# Define the complete workflow
def create_workflow():
    """Create and return the LangGraph workflow."""
    # Create the graph with type hints
    workflow = StateGraph(QueryState)
    
    # Add nodes
    workflow.add_node("load_prompts", load_prompts)
    workflow.add_node("match_prompt", match_prompt)
    workflow.add_node("enhance", enhance_query)
    workflow.add_node("validate", validate_query)
    workflow.add_node("adjust", adjust_query)
    workflow.add_node("generate", generate_final_result)
    
    # Define edges
    workflow.add_edge(START, "load_prompts")
    workflow.add_edge("load_prompts", "match_prompt")
    workflow.add_conditional_edges(
        "match_prompt",
        route_after_match,
        {
            "enhance": "enhance",
            "generate": "generate"
        }
    )
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
    Optimize a user query by applying the appropriate template from MLflow using LangGraph workflow.
    
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
            "prompt_match": result.get("prompt_match", {"status": "unknown"}),
            "content_type": result.get("content_type"),
            "enhanced": not result.get("should_skip_enhance", False),
            "initial_enhanced_query": result.get("enhanced_query", query) 
                if result.get("validation_result") == "NEEDS_ADJUSTMENT" and not result.get("should_skip_enhance", False) 
                else None,
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
            "enhanced": False,
            "error": str(e)
        }

@mcp.tool()
async def list_prompts() -> Dict[str, Any]:
    """
    List all available prompts from MLflow Prompt Registry.
    
    Returns:
        List of prompts and their metadata
    """
    try:
        setup_mlflow_connection()
        prompts_dict = load_all_prompts()
        
        # Convert to a list of metadata for client display
        prompts_list = []
        for name, prompt in prompts_dict.items():
            # Extract content type from name (e.g., "essay_prompt" -> "essay")
            content_type = name.replace("_prompt", "")
            
            # Get variables from template
            import re
            variables = []
            for match in re.finditer(r'{{([^{}]+)}}', prompt.template):
                var_name = match.group(1).strip()
                variables.append(var_name)
            
            prompts_list.append({
                "name": name,
                "type": content_type,
                "version": prompt.version,
                "variables": variables,
                "tags": getattr(prompt, "tags", {})
            })
        
        return {
            "prompts": prompts_list,
            "count": len(prompts_list)
        }
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        return {
            "error": str(e),
            "prompts": [],
            "count": 0
        }

if __name__ == "__main__":
    logger.info("Starting PromptLab server with MLflow Prompt Registry")
    try:
        # Ensure MLflow connection is set up
        setup_mlflow_connection()
        
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)