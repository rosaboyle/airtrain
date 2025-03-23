# AirTrain Agent System

This package provides a comprehensive agent system for building AI applications with memory management and tool integration capabilities.

## Key Components

### Memory System

The memory system allows agents to maintain both short-term and long-term memory:

1. **BaseMemory** - Core base class for all memory types
2. **ShortTermMemory** - Memory with automatic summarization when exceeding message limits
3. **LongTermMemory** - Persistent memory with keyword extraction and retrieval capabilities
4. **SharedMemory** - Memory that can be shared between multiple agents
5. **AgentMemoryManager** - Manages multiple memory instances for an agent

### Agent Registry

The agent registry provides a way to register and instantiate agent classes:

1. **BaseAgent** - Core base class for all agents
2. **AgentRegistry** - Registration system for agent classes
3. **AgentFactory** - Factory for creating agent instances
4. **register_agent** - Decorator for registering agent classes

## Usage Examples

### Creating a Custom Agent

```python
from airtrain.agents import BaseAgent, register_agent
from airtrain.tools import ToolFactory

@register_agent("my_custom_agent")
class MyCustomAgent(BaseAgent):
    """An example custom agent implementation."""
    
    def __init__(self, name, models=None, tools=None):
        super().__init__(name, models, tools)
        # Initialize custom memory
        self.create_memory("dialog", 15)
        
    def process(self, user_input, memory_name="dialog"):
        """Process user input and return a response."""
        # Add to memory
        self.memory.add_to_all({"role": "user", "content": user_input})
        
        # Implement your custom processing logic here
        response = f"Echo: {user_input}"
        
        # Add response to memory
        self.memory.add_to_all({"role": "assistant", "content": response})
        
        return response
```

### Instantiating an Agent

```python
from airtrain.agents import AgentFactory
from airtrain.tools import ToolFactory

# Get a tool
calculator = ToolFactory.get_tool("calculator")

# Create an agent instance
agent = AgentFactory.create_agent(
    agent_type="conversation_agent",
    name="MyAssistant",
    models=["llama-3.1-8b-instant"],
    tools=[calculator]
)

# Process user input
response = agent.process("Hello, can you help me calculate 23 * 17?")
print(response)
```

### Creating an Agent Team with Shared Memory

```python
from airtrain.agents import AgentFactory
from airtrain.agents.memory import SharedMemory

# Create shared memory
team_memory = SharedMemory("team_knowledge")

# Create agents
agent1 = AgentFactory.create_agent("conversation_agent", name="Agent1")
agent2 = AgentFactory.create_agent("conversation_agent", name="Agent2")

# Add shared memory to both agents
agent1.memory.add_shared_memory(team_memory)
agent2.memory.add_shared_memory(team_memory)

# Now both agents can access the same shared memory
agent1.process("The password is 12345")
agent2.process("What was the password?")  # Agent2 can access info from shared memory
```

## Key Concepts

### Memory Management

The agent system provides sophisticated memory management:

- **Short-term Memory** - Automatically summarizes older messages when exceeding limits
- **Long-term Memory** - Extracts keywords and can be persisted to disk
- **Multiple Memory Contexts** - Agents can maintain multiple specialized memories (e.g., dialog vs. reasoning)
- **Shared Memory** - Enables collaboration between multiple agents

### Tool Integration

Agents can use tools from the AirTrain tools registry:

```python
# Add a tool to an agent
agent.add_tool(ToolFactory.get_tool("calculator"))

# Register multiple tools
agent.register_tools([
    ToolFactory.get_tool("calculator"),
    ToolFactory.get_tool("conversation_memory", "stateful")
])
```

### LLM Backend Support

The example agent implementation supports multiple LLM backends:

- Groq
- Fireworks
- More can be added by implementing similar adapters

## Best Practices

1. **Separate Dialog from Reasoning** - Use different memory contexts for conversation vs. reasoning
2. **Leverage Shared Memory** - Use shared memory for multi-agent systems that need to collaborate
3. **Persist Long-term Memory** - Save important information using the persistence capabilities
4. **Use Appropriate Tools** - Select tools that complement your agent's purpose

For more detailed examples, see the `example_agent.py` module. 