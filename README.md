# PromptLab: AI Query Enhancement Engine

PromptLab transforms basic user queries into optimized prompts for AI systems. It automatically detects content type (essays, emails, technical explanations, creative writing), applies tailored templates, and validates that enhanced prompts maintain the original intent.

## 🔍 Overview

PromptLab is built on a modular architecture with a YAML-based template system that enables anyone to create and manage prompt templates without coding knowledge. The system ultimately produces higher-quality AI responses through better-structured inputs.


## 🏗️ Architecture

PromptLab consists of three primary components:

1. **Template System** (`prompt_templates.yaml`) - Structured templates for different content types
2. **MCP Server** (`promptlab_server.py`) - Serves templates through a standardized protocol
3. **Processing Client** (`promptlab_client.py`) - LangGraph workflow that handles the transformation process

### Workflow Process

1. **Query Input**: User submits a natural language query
2. **Classification**: System determines the content type (essay, email, etc.)
3. **Parameter Extraction**: Key parameters are identified (topic, audience, etc.)
4. **Template Application**: The appropriate template is retrieved and filled
5. **Validation**: The enhanced prompt is checked against the original intent
6. **Adjustment**: Any needed refinements are made automatically
7. **Response Generation**: The optimized prompt produces a high-quality response

![alt text](<promptlab_workflow.png>)

## 📋 Features

- **Content Type Detection** - Automatically classifies user queries into essay, email, technical, or creative writing requests
- **Parameter Extraction** - Intelligently extracts key parameters like topics, recipients, and audience levels
- **Template Library** - Pre-configured templates for common content types with structured guidance
- **Validation System** - Ensures enhanced prompts maintain the original user intent
- **Feedback Loop** - Adjusts prompts when validation identifies misalignments
- **Modular Design** - MCP server can be plugged into any LLM system
- **Non-Technical Management** - Templates can be updated without coding knowledge

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Dependencies:
  - `mcp[cli]`
  - `langchain-openai`
  - `langgraph>=0.0.20`
  - `python-dotenv`
  - `pyyaml`

### Installation

```bash
# Clone the repository
git clone https://github.com/iRahulPandey/PromptLab.git
cd PromptLab

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env
# Edit .env to add your OpenAI API key
```

### Usage

1. Start by running the server:

```bash
# The server loads templates from prompt_templates.yaml
python promptlab_server.py
```

2. Run the client with your query:

```bash
python promptlab_client.py "Write an essay about climate change"
```

3. The system will output:
   - Original query
   - Classified content type
   - Enhanced prompt
   - Validation result
   - Final response

## 📝 Template System

Templates are defined in `prompt_templates.yaml` using a structured format:

```yaml
templates:
  essay_prompt:
    description: "Generate an optimized prompt template for writing essays."
    template: |
      Write a well-structured essay on {topic} that includes:
      - A compelling introduction that provides context and states your thesis
      ...
    parameters:
      - name: topic
        type: string
        description: The topic of the essay
        required: true
```

### Adding New Templates

1. Open `prompt_templates.yaml`
2. Add a new template following the existing format
3. Define parameters and transformations
4. Define a tool on server side and load the template
5. The server will automatically load the new template on restart

## 🛠️ Advanced Configuration

### Environment Variables

- `TEMPLATES_FILE` - Path to the templates YAML file (default: `prompt_templates.yaml`)
- `OPENAI_API_KEY` - Your OpenAI API key for LLM access
- `MODEL_NAME` - The OpenAI model to use (default: `gpt-3.5-turbo`)
- `PERSONA_SERVER_SCRIPT` - Path to the server script (default: `promptlab_server.py`)

### Custom Transformations

Templates can include transformations that dynamically adjust parameters:

```yaml
transformations:
  - name: formality
    value: "formal if recipient_type.lower() in ['boss', 'client'] else 'semi-formal'"
```

## 📊 Example Outputs

### Input Query
> "Write something about renewable energy for my professor"

### Enhanced Prompt
```
Write a well-structured essay on renewable energy that includes:
- A compelling introduction that provides context and states your thesis
- 2-3 body paragraphs, each with a clear topic sentence and supporting evidence
- Logical transitions between paragraphs that guide the reader
- A conclusion that synthesizes your main points and offers final thoughts

The essay should be informative, well-reasoned, and demonstrate critical thinking.
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
