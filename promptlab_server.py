from mcp.server.fastmcp import FastMCP
import logging
import sys
import yaml
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promptlab-server")

# File paths
TEMPLATES_FILE = os.environ.get("TEMPLATES_FILE", "prompt_templates.yaml")

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

def evaluate_transformation(transformation, params):
    """Evaluate a transformation expression using the provided parameters."""
    # Very simple expression evaluator - for production, consider a safer approach
    expr = transformation
    for param_name, param_value in params.items():
        # Replace parameter references in the expression
        if isinstance(param_value, str):
            expr = expr.replace(param_name, f"'{param_value}'")
        else:
            expr = expr.replace(param_name, str(param_value))
    
    try:
        # Handle "value if condition else other_value" syntax
        if " if " in expr and " else " in expr:
            return eval(expr)
        return eval(expr)
    except Exception as e:
        logger.error(f"Error evaluating transformation '{transformation}': {e}")
        return None

def apply_template(template_config, args):
    """Apply a template with the given arguments and transformations."""
    try:
        # Get the template string
        template_str = template_config['template']
        
        # Process any transformations
        if 'transformations' in template_config:
            for transform in template_config['transformations']:
                transform_name = transform['name']
                transform_expr = transform['value']
                args[transform_name] = evaluate_transformation(transform_expr, args)
        
        # Format the template with the arguments
        return template_str.format(**args)
    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return f"Error: Missing required parameter: {e}"
    except Exception as e:
        logger.error(f"Error applying template: {e}")
        return f"Error applying template: {e}"

# Initialize MCP server
mcp = FastMCP(
    name="persona",
    instructions="I provide specialized templates for enhancing prompts based on content type."
)

# Load templates
templates = load_templates(TEMPLATES_FILE)
if not templates:
    logger.error(f"No templates found in {TEMPLATES_FILE}. Server will not be able to provide templates.")

# Register fixed set of template tools
@mcp.tool()
def essay_prompt(topic: str) -> str:
    """
    Generate an optimized prompt template for writing essays.
    
    Args:
        topic: The topic of the essay
    
    Returns:
        An enhanced prompt template for writing an essay
    """
    logger.info(f"Generating essay template for topic: {topic}")
    if "essay_prompt" in templates:
        return apply_template(templates["essay_prompt"], {"topic": topic})
    else:
        return f"Write a well-structured essay about {topic}."

@mcp.tool()
def email_prompt(recipient_type: str, topic: str) -> str:
    """
    Generate an optimized prompt template for writing emails.
    
    Args:
        recipient_type: The type of recipient (e.g., boss, colleague, client)
        topic: The subject/purpose of the email
    
    Returns:
        An enhanced prompt template for writing an email
    """
    logger.info(f"Generating email template for recipient: {recipient_type}, topic: {topic}")
    if "email_prompt" in templates:
        return apply_template(templates["email_prompt"], {
            "recipient_type": recipient_type,
            "topic": topic
        })
    else:
        return f"Write an email to my {recipient_type} about {topic}."

@mcp.tool()
def technical_prompt(topic: str, audience: str = "general") -> str:
    """
    Generate an optimized prompt template for technical explanations.
    
    Args:
        topic: The technical topic to explain
        audience: Target audience knowledge level (beginner, intermediate, expert, or general)
    
    Returns:
        An enhanced prompt template for technical explanations
    """
    logger.info(f"Generating technical template for topic: {topic}, audience: {audience}")
    if "technical_prompt" in templates:
        return apply_template(templates["technical_prompt"], {
            "topic": topic,
            "audience": audience
        })
    else:
        return f"Provide a technical explanation of {topic} for a {audience} audience."

@mcp.tool()
def creative_prompt(genre: str, topic: str) -> str:
    """
    Generate an optimized prompt template for creative writing.
    
    Args:
        genre: The creative writing genre (story, poem, script, etc.)
        topic: The theme or subject of the creative piece
    
    Returns:
        An enhanced prompt template for creative writing
    """
    logger.info(f"Generating creative template for genre: {genre}, topic: {topic}")
    if "creative_prompt" in templates:
        return apply_template(templates["creative_prompt"], {
            "genre": genre,
            "topic": topic
        })
    else:
        return f"Write a creative {genre} about {topic}."

if __name__ == "__main__":
    logger.info(f"Starting Persona MCP server with templates from {TEMPLATES_FILE}")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)