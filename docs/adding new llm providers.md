

## **🎯 Minimal Changes Needed for Gemini Support**

### **1. Add Gemini Model Creation Method (NEW METHOD)**

Add this new method to your existing `AgentFactory` class:

```python
# agents/agents.py - ADD Gemini support method
async def _get_gemini_model(self, llm_node: str, node_config: dict) -> Any:
    """Get Gemini model for function calling"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        model = ChatGoogleGenerativeAI(
            model=node_config.get('model', 'gemini-pro'),
            temperature=node_config.get('temperature', 0.7),
            max_tokens=node_config.get('max_tokens', 1500),
        )
        
        # Test the model
        test_response = await model.ainvoke([HumanMessage(content="test")])
        logger.debug(f"✅ Gemini model {llm_node} working")
        
        return model
        
    except Exception as e:
        logger.warning(f"⚠️ Gemini model failed, falling back to LLM manager: {e}")
        return await llm_manager._get_model(llm_node)
```


### **2. Update Existing Provider Detection (SMALL UPDATE)**

Update your existing `_get_agent_llm` method to include Gemini:

```python
# agents/agents.py - UPDATE existing method (just add elif)
async def _get_agent_llm(self, llm_node: str) -> Any:
    """Get LLM instance for agent with provider awareness"""
    try:
        # Get node configuration to determine provider
        node_config = config.llm_config['nodes'].get(llm_node, {})
        provider = node_config.get('provider', 'anthropic')
        
        # 🔥 EXISTING: OpenAI support
        if provider == 'openai':
            return await self._get_openai_model(llm_node, node_config)
        # 🔥 EXISTING: Anthropic support  
        elif provider == 'anthropic':
            return await self._get_anthropic_model(llm_node)
        # 🔥 NEW: Just add this elif for Gemini
        elif provider == 'gemini':
            return await self._get_gemini_model(llm_node, node_config)
        else:
            # Fallback to LLM manager for other providers
            return await llm_manager._get_model(llm_node)
            
    except Exception as e:
        logger.error(f"Failed to get LLM for node {llm_node}: {e}")
        raise
```


### **3. Add Gemini Provider Configuration (CONFIG ONLY)**

Add to your modular provider system:

```yaml
# config/providers/gemini.yaml - NEW FILE
provider_info:
  name: "gemini"
  display_name: "Google Gemini"
  type: "gemini"
  description: "Google Gemini models with function calling"
  documentation: "https://developers.generativeai.google/"

connection:
  api_key_env: "GOOGLE_API_KEY"
  base_url: "https://generativelanguage.googleapis.com"
  timeout: 30
  max_retries: 3

capabilities:
  function_calling: true
  streaming: true
  vision: true
  embeddings: false
  fine_tuning: false

rate_limits:
  requests_per_minute: 60  # Free tier limit
  tokens_per_minute: 32000
  max_concurrent_requests: 2

models:
  gemini-pro:
    display_name: "Gemini Pro"
    max_tokens: 8192
    supports_functions: true
    cost_per_1k_tokens: 0.0  # Currently free
    
  gemini-pro-vision:
    display_name: "Gemini Pro Vision"
    max_tokens: 4096
    supports_functions: true
    supports_vision: true
    cost_per_1k_tokens: 0.0

node_templates:
  chat_default:
    model: "gemini-pro"
    temperature: 0.7
    max_tokens: 1500
    
  coder_default:
    model: "gemini-pro"
    temperature: 0.2
    max_tokens: 2000
```

```yaml
# config/providers/registry.yaml - UPDATE existing file
active_providers:
  - openai
  - anthropic
  - gemini  # Just add this line

# Rest stays the same...
```


### **4. Update Main Configuration (SMALL UPDATE)**

```yaml
# config/llm_config.yaml - ADD Gemini nodes
nodes:
  chat:
    provider: "openai"    # Existing
    template: "chat_default"
    
  coder:
    provider: "openai"    # Existing  
    template: "coder_default"
    
  # 🔥 NEW: Add Gemini nodes
  gemini_chat:
    provider: "gemini"
    template: "chat_default"
    
  gemini_coder:
    provider: "gemini"
    template: "coder_default"
```


### **5. Install Dependencies (ONE LINE)**

```bash
pip install langchain-google-genai
```



### **✅ What You're Adding:**

- One new method (`_get_gemini_model`)
- One new line in existing method (`elif provider == 'gemini'`)
- Configuration files (no code changes)


## **🎯 Future Provider Support**

With this pattern, adding **any new provider** (Claude, Azure OpenAI, local models, etc.) follows the same pattern:

```python
# Future: Add any provider with just these steps
# 1. Add _get_[provider]_model method
# 2. Add elif provider == '[provider]' to _get_agent_llm  
# 3. Add provider config files
# 4. Install dependencies
```


## **🔧 Optional: Provider-Specific Optimizations**

You could also extend your optional provider-specific agent creation:

```python
# agents/agents.py - OPTIONAL enhancement
async def _create_gemini_tool_agent(self, llm, agent_tools, prompt, agent_config):
    """Create Gemini-optimized tool agent"""
    try:
        # Use generic tool calling - Gemini supports LangChain standard
        agent = create_tool_calling_agent(llm, agent_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=agent_tools,
            max_iterations=agent_config.max_iterations,
            verbose=agent_config.verbose,
            return_intermediate_steps=True
        )
    except Exception as e:
        logger.warning(f"Gemini tool agent creation failed: {e}")
        # Fallback to simple chain
        return prompt | llm
```
