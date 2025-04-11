#!/usr/bin/env python
import os
import argparse
import logging
import json
import mlflow
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow-prompt-registry")

# Set up environment variables
from dotenv import load_dotenv
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')

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

def register_prompt(
    name: str, 
    template: str, 
    commit_message: str = "Initial commit", 
    tags: Optional[Dict[str, str]] = None, 
    version_metadata: Optional[Dict[str, str]] = None,
    set_as_production: bool = True
) -> Dict[str, Any]:
    """
    Register a prompt in MLflow Prompt Registry.
    
    Args:
        name: Name of the prompt
        template: Template text with variables in {{ variable }} format
        commit_message: Description of the prompt or changes
        tags: Optional key-value pairs for categorization
        version_metadata: Optional metadata for this prompt version
        set_as_production: Whether to set this version as the production alias
        
    Returns:
        Dictionary with registration details
    """
    try:
        # Check if the prompt already exists with a production alias
        previous_production_version = None
        try:
            previous_prompt = mlflow.load_prompt(f"prompts:/{name}@production")
            previous_production_version = previous_prompt.version
            logger.info(f"Found existing production version {previous_production_version} for '{name}'")
        except Exception:
            logger.info(f"No existing production version found for '{name}'")
        
        # Register the prompt
        prompt = mlflow.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags=tags or {},
            version_metadata=version_metadata or {}
        )
        
        # Handle aliasing
        if set_as_production:
            # Archive the previous production version if it exists
            if previous_production_version is not None:
                mlflow.set_prompt_alias(name, "archived", previous_production_version)
                logger.info(f"Archived '{name}' version {previous_production_version}")
                
            # Set new version as production
            mlflow.set_prompt_alias(name, "production", prompt.version)
            logger.info(f"Set '{name}' version {prompt.version} as production alias")
        
        result = {
            "name": name,
            "version": prompt.version,
            "status": "success",
            "production": set_as_production
        }
        
        # Add archived information if applicable
        if previous_production_version is not None:
            result["previous_production"] = previous_production_version
            result["archived"] = True
            
        return result
    except Exception as e:
        logger.error(f"Failed to register prompt '{name}': {e}")
        return {
            "name": name,
            "status": "error",
            "error": str(e)
        }

def register_from_file(file_path: str, set_as_production: bool = True) -> Dict[str, Any]:
    """
    Register prompts from a JSON file.
    
    The JSON file should have the format:
    {
        "prompts": [
            {
                "name": "prompt_name",
                "template": "Template text with {{ variables }}",
                "commit_message": "Description",
                "tags": {"key": "value"},
                "version_metadata": {"author": "name"}
            }
        ]
    }
    
    Args:
        file_path: Path to the JSON file
        set_as_production: Whether to set these versions as production aliases
        
    Returns:
        Dictionary with registration results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError("JSON file must contain a 'prompts' list")
        
        results = []
        for prompt_data in data["prompts"]:
            name = prompt_data.get("name")
            if not name:
                logger.warning("Skipping prompt without name")
                continue
                
            template = prompt_data.get("template")
            if not template:
                logger.warning(f"Skipping prompt '{name}' without template")
                continue
            
            result = register_prompt(
                name=name,
                template=template,
                commit_message=prompt_data.get("commit_message", "Registered from file"),
                tags=prompt_data.get("tags"),
                version_metadata=prompt_data.get("version_metadata"),
                set_as_production=set_as_production
            )
            results.append(result)
        
        return {
            "status": "success",
            "file": file_path,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to register prompts from file '{file_path}': {e}")
        return {
            "status": "error",
            "file": file_path,
            "error": str(e)
        }

def register_sample_prompts() -> Dict[str, Any]:
    """
    Register standard sample prompts for each content type.
    
    Returns:
        Dictionary with registration results
    """
    results = []
    
    # Essay prompt
    essay_result = register_prompt(
        name="essay_prompt",
        template="""
        Write a well-structured essay on {{ topic }} that includes:
        - A compelling introduction that provides context and states your thesis
        - 2-3 body paragraphs, each with a clear topic sentence and supporting evidence
        - Logical transitions between paragraphs that guide the reader
        - A conclusion that synthesizes your main points and offers final thoughts
        
        The essay should be informative, well-reasoned, and demonstrate critical thinking.
        """,
        commit_message="Initial essay prompt",
        tags={"task": "writing", "type": "essay"}
    )
    results.append(essay_result)
    
    # Email prompt
    email_result = register_prompt(
        name="email_prompt",
        template="""
        Write a {{ formality }} email to my {{ recipient_type }} about {{ topic }} that includes:
        - A clear subject line
        - Appropriate greeting
        - Brief introduction stating the purpose
        - Main content in short paragraphs
        - Specific action items or requests clearly highlighted
        - Professional closing
        
        The tone should be {{ tone }}.
        """,
        commit_message="Initial email prompt",
        tags={"task": "writing", "type": "email"}
    )
    results.append(email_result)
    
    # Technical prompt
    technical_result = register_prompt(
        name="technical_prompt",
        template="""
        Provide a clear technical explanation of {{ topic }} for a {{ audience }} audience that:
        - Begins with a conceptual overview that anyone can understand
        - Uses analogies or real-world examples to illustrate complex concepts
        - Defines technical terminology when first introduced
        - Gradually increases in technical depth
        - Includes practical applications or implications where relevant
        - Addresses common misunderstandings or misconceptions
        """,
        commit_message="Initial technical prompt",
        tags={"task": "explanation", "type": "technical"}
    )
    results.append(technical_result)
    
    # Creative prompt
    creative_result = register_prompt(
        name="creative_prompt",
        template="""
        Write a creative {{ genre }} about {{ topic }} that:
        - Uses vivid sensory details and imagery
        - Develops interesting and multidimensional characters (if applicable)
        - Creates an engaging narrative arc with tension and resolution
        - Establishes a distinct mood, tone, and atmosphere
        - Employs figurative language to enhance meaning
        - Avoids clichés and predictable elements
        """,
        commit_message="Initial creative prompt",
        tags={"task": "writing", "type": "creative"}
    )
    results.append(creative_result)
    
    return {
        "status": "success",
        "registered": len(results),
        "results": results
    }

def list_prompts() -> Dict[str, Any]:
    """
    List all prompts in the MLflow Prompt Registry.
    
    Returns:
        Dictionary with prompt information
    """
    try:
        # Standard content types
        content_types = ["essay", "email", "technical", "creative"]
        # Custom types we'll look for
        custom_types = ["code", "summary", "analysis", "qa", "custom", "social_media", 
                       "blog", "report", "letter", "presentation", "review", "comparison", 
                       "instruction"]
        # Combined list for checking
        all_types = content_types + custom_types
        
        prompts = []
        
        # Check for standard and custom prompt types
        for content_type in all_types:
            prompt_name = f"{content_type}_prompt"
            try:
                # Try different alias approaches to get as much information as possible
                production_version = None
                archived_version = None
                latest_prompt = None
                
                # Try to get production version
                try:
                    production_prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@production")
                    production_version = production_prompt.version
                    latest_prompt = production_prompt  # Use production as latest if available
                except Exception:
                    pass
                
                # Try to get archived version
                try:
                    archived_prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@archived")
                    archived_version = archived_prompt.version
                except Exception:
                    pass
                
                # If we don't have a production version, try to get latest
                if latest_prompt is None:
                    try:
                        latest_prompt = mlflow.load_prompt(f"prompts:/{prompt_name}")
                    except Exception:
                        continue  # Skip if we can't get any version
                
                # Add prompt information
                prompt_info = {
                    "name": prompt_name,
                    "type": content_type,
                    "latest_version": latest_prompt.version,
                    "production_version": production_version,
                    "archived_version": archived_version,
                    "tags": getattr(latest_prompt, "tags", {})
                }
                
                prompts.append(prompt_info)
            except Exception as e:
                # Skip if prompt doesn't exist or can't be loaded
                logger.debug(f"Could not load prompt '{prompt_name}': {e}")
        
        # Look for any other prompts that don't follow the standard pattern
        # This would require tracking prompt names separately or using MLflow's API
        # to list all registered models (prompts) if such functionality becomes available
        
        return {
            "status": "success",
            "prompts": prompts,
            "count": len(prompts)
        }
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prompts": []
        }

def update_prompt(name: str, template: str, commit_message: str, set_as_production: bool = True) -> Dict[str, Any]:
    """
    Update an existing prompt with a new version.
    
    Args:
        name: Name of the prompt to update
        template: New template text
        commit_message: Description of the changes
        set_as_production: Whether to set this version as the production alias
        
    Returns:
        Dictionary with update details
    """
    try:
        # Check if the prompt exists
        previous_version = None
        previous_production_version = None
        
        # Try to get the latest version
        try:
            previous_prompt = mlflow.load_prompt(f"prompts:/{name}")
            previous_version = previous_prompt.version
        except Exception as e:
            return {
                "name": name,
                "status": "error",
                "error": f"Prompt '{name}' not found: {str(e)}"
            }
        
        # Try to get the production version
        try:
            production_prompt = mlflow.load_prompt(f"prompts:/{name}@production")
            previous_production_version = production_prompt.version
        except:
            logger.info(f"No production alias found for '{name}'")
        
        # Register a new version
        prompt = mlflow.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message
        )
        
        # Handle aliasing
        if set_as_production:
            # Archive the previous production version if it exists
            if previous_production_version is not None:
                mlflow.set_prompt_alias(name, "archived", previous_production_version)
                logger.info(f"Archived '{name}' version {previous_production_version}")
                
            # Set new version as production
            mlflow.set_prompt_alias(name, "production", prompt.version)
            logger.info(f"Set '{name}' version {prompt.version} as production alias")
        
        result = {
            "name": name,
            "previous_version": previous_version,
            "new_version": prompt.version,
            "status": "success",
            "production": set_as_production
        }
        
        # Add archived information if applicable
        if previous_production_version is not None:
            result["previous_production"] = previous_production_version
            result["archived"] = previous_production_version != prompt.version
            
        return result
    except Exception as e:
        logger.error(f"Failed to update prompt '{name}': {e}")
        return {
            "name": name,
            "status": "error",
            "error": str(e)
        }

def get_prompt_details(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a prompt and all its versions.
    
    Args:
        name: Name of the prompt
        
    Returns:
        Dictionary with prompt details
    """
    try:
        # Try to get production version
        production_version = None
        production_template = None
        try:
            production_prompt = mlflow.load_prompt(f"prompts:/{name}@production")
            production_version = production_prompt.version
            production_template = production_prompt.template
        except:
            pass
            
        # Try to get archived version
        archived_versions = []
        try:
            archived_prompt = mlflow.load_prompt(f"prompts:/{name}@archived")
            archived_versions.append(archived_prompt.version)
        except:
            pass
            
        # Try to get latest version
        latest_version = None
        latest_template = None
        latest_tags = None
        try:
            latest_prompt = mlflow.load_prompt(f"prompts:/{name}")
            latest_version = latest_prompt.version
            latest_template = latest_prompt.template
            latest_tags = getattr(latest_prompt, "tags", {})
        except Exception as e:
            return {
                "name": name,
                "status": "error",
                "error": f"Prompt '{name}' not found: {str(e)}"
            }
            
        # Extract variables from the template
        variables = []
        import re
        for match in re.finditer(r'{{([^{}]+)}}', latest_template):
            var_name = match.group(1).strip()
            variables.append(var_name)
            
        return {
            "name": name,
            "status": "success",
            "latest_version": latest_version,
            "production_version": production_version,
            "archived_versions": archived_versions,
            "variables": variables,
            "tags": latest_tags,
            "latest_template": latest_template,
            "production_template": production_template if production_version != latest_version else None
        }
    except Exception as e:
        logger.error(f"Failed to get details for prompt '{name}': {e}")
        return {
            "name": name,
            "status": "error",
            "error": str(e)
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="MLflow Prompt Registry Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new prompt")
    register_parser.add_argument("--name", required=True, help="Name of the prompt")
    register_parser.add_argument("--template", required=True, help="Template text or path to template file")
    register_parser.add_argument("--message", default="Initial commit", help="Commit message")
    register_parser.add_argument("--tags", help="Tags in JSON format (e.g., '{\"task\": \"writing\"}')")
    register_parser.add_argument("--metadata", help="Version metadata in JSON format")
    register_parser.add_argument("--no-production", action="store_true", help="Don't set as production alias")
    
    # Register from file command
    file_parser = subparsers.add_parser("register-file", help="Register prompts from a JSON file")
    file_parser.add_argument("--file", required=True, help="Path to JSON file with prompts")
    file_parser.add_argument("--no-production", action="store_true", help="Don't set as production alias")
    
    # Register samples command
    subparsers.add_parser("register-samples", help="Register sample prompts for common content types")
    
    # List command
    subparsers.add_parser("list", help="List all prompts in the registry")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update an existing prompt")
    update_parser.add_argument("--name", required=True, help="Name of the prompt to update")
    update_parser.add_argument("--template", required=True, help="New template text or path to template file")
    update_parser.add_argument("--message", required=True, help="Commit message describing the changes")
    update_parser.add_argument("--no-production", action="store_true", help="Don't set as production alias")
    
    # Get details command
    details_parser = subparsers.add_parser("details", help="Get detailed information about a prompt")
    details_parser.add_argument("--name", required=True, help="Name of the prompt")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup MLflow connection
    setup_mlflow_connection()
    
    # Process command
    if args.command == "register":
        # Check if template is a file path
        template = args.template
        if os.path.isfile(template):
            with open(template, 'r') as f:
                template = f.read()
        
        # Parse tags and metadata if provided
        tags = None
        if args.tags:
            try:
                tags = json.loads(args.tags)
            except:
                logger.error("Invalid JSON for tags")
                return 1
        
        version_metadata = None
        if args.metadata:
            try:
                version_metadata = json.loads(args.metadata)
            except:
                logger.error("Invalid JSON for metadata")
                return 1
        
        result = register_prompt(
            name=args.name,
            template=template,
            commit_message=args.message,
            tags=tags,
            version_metadata=version_metadata,
            set_as_production=not args.no_production
        )
        
        print(json.dumps(result, indent=2))
    
    elif args.command == "register-file":
        result = register_from_file(
            file_path=args.file,
            set_as_production=not args.no_production
        )
        
        print(json.dumps(result, indent=2))
    
    elif args.command == "register-samples":
        result = register_sample_prompts()
        print(json.dumps(result, indent=2))
    
    elif args.command == "list":
        result = list_prompts()
        print(json.dumps(result, indent=2))
    
    elif args.command == "update":
        # Check if template is a file path
        template = args.template
        if os.path.isfile(template):
            with open(template, 'r') as f:
                template = f.read()
                
        result = update_prompt(
            name=args.name,
            template=template,
            commit_message=args.message,
            set_as_production=not args.no_production
        )
        
        print(json.dumps(result, indent=2))
        
    elif args.command == "details":
        result = get_prompt_details(args.name)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())