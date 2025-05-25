import asyncio
import os
import time
from typing import Dict, Any, List, Optional
from anthropic import Anthropic
from tavily import TavilyClient
import requests
from urllib.parse import urlparse

class WebAgent:
    """Web browsing and search agent for Mortey using Tavily API"""
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize Tavily client
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
    async def search_and_browse(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main method for web search and browsing using Tavily"""
        
        user_input = state.get('user_input', '')
        
        # Update thinking state
        state['thinking_state'] = {
            'active_agent': 'WEB',
            'current_task': 'Analyzing web request',
            'progress': 0.1,
            'details': f'Processing: {user_input}'
        }
        
        try:
            # Analyze the request to determine action
            action_plan = await self._analyze_web_request(user_input)
            
            state['thinking_state'] = {
                'active_agent': 'WEB',
                'current_task': 'Searching the web',
                'progress': 0.4,
                'details': f'Searching for: {action_plan.get("query", user_input)}'
            }
            
            results = []
            
            if action_plan['action'] in ['search', 'search_and_visit']:
                # Use Tavily for search
                results = await self._tavily_search(action_plan['query'])
                
            elif action_plan['action'] == 'visit':
                # For specific URL visits, we can still get content
                results = await self._get_url_content(action_plan['url'])
            
            # Generate summary using Claude
            state['thinking_state'] = {
                'active_agent': 'WEB',
                'current_task': 'Generating summary',
                'progress': 0.8,
                'details': f'Analyzing {len(results)} results'
            }
            
            summary = await self._generate_summary(user_input, results)
            
            return {
                **state,
                'web_results': results,
                'output_content': summary,
                'output_type': 'web_results',
                'thinking_state': {
                    'active_agent': 'WEB',
                    'current_task': 'Web search complete',
                    'progress': 1.0,
                    'details': f'Found {len(results)} results'
                }
            }
            
        except Exception as e:
            return {
                **state,
                'web_results': [],
                'output_content': f"Web search error: {str(e)}",
                'output_type': 'error',
                'thinking_state': {
                    'active_agent': 'WEB',
                    'current_task': 'Error occurred',
                    'progress': 1.0,
                    'details': f'Error: {str(e)}'
                }
            }
    
    async def _analyze_web_request(self, user_input: str) -> Dict[str, Any]:
        """Use Claude to analyze what web action to take"""
        
        analysis_prompt = f"""
        Analyze this user request and determine the appropriate web action:
        
        User Request: {user_input}
        
        Determine:
        1. Action type: "search", "visit", or "search_and_visit"
        2. Search query (if needed) - optimize for web search
        3. URL (if specific website mentioned)
        
        Respond in JSON format:
        {{
            "action": "search/visit/search_and_visit",
            "query": "optimized search terms if needed",
            "url": "specific URL if mentioned"
        }}
        """
        
        try:
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            import json
            response_text = message.content[0].text
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
            else:
                # Fallback to simple search
                return {
                    "action": "search",
                    "query": user_input
                }
                
        except Exception as e:
            # Fallback to simple search
            return {
                "action": "search", 
                "query": user_input
            }
    
    async def _tavily_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform search using Tavily API"""
        
        try:
            # Use Tavily's search with content extraction
            search_response = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                search_depth="advanced",  # Get more detailed results
                include_answer=True,      # Get AI-generated answer
                include_raw_content=True, # Get full content
                max_results=max_results
            )
            
            results = []
            
            # Add the AI-generated answer if available
            if search_response.get('answer'):
                results.append({
                    "title": "AI Summary",
                    "url": "",
                    "content": search_response['answer'],
                    "snippet": search_response['answer'][:300] + "...",
                    "source": "tavily_ai",
                    "type": "ai_answer"
                })
            
            # Add search results
            for result in search_response.get('results', []):
                results.append({
                    "title": result.get('title', 'Unknown Title'),
                    "url": result.get('url', ''),
                    "content": result.get('content', ''),
                    "snippet": result.get('content', '')[:300] + "..." if result.get('content') else '',
                    "source": "tavily_search",
                    "type": "search_result",
                    "score": result.get('score', 0)
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Tavily search failed: {str(e)}")
    
    async def _get_url_content(self, url: str) -> List[Dict[str, Any]]:
        """Get content from a specific URL using Tavily"""
        
        try:
            # Use Tavily to extract content from URL
            content_response = await asyncio.to_thread(
                self.tavily_client.extract,
                urls=[url]
            )
            
            results = []
            
            for result in content_response.get('results', []):
                results.append({
                    "title": result.get('title', 'Unknown Title'),
                    "url": result.get('url', url),
                    "content": result.get('content', ''),
                    "snippet": result.get('content', '')[:300] + "..." if result.get('content') else '',
                    "source": "tavily_extract",
                    "type": "url_content"
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"URL content extraction failed: {str(e)}")
    
    async def _generate_summary(self, user_request: str, results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive summary of web results"""
        
        # Prepare results for Claude
        results_text = ""
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 for cost control
            results_text += f"\n{i}. {result.get('title', 'Unknown')}\n"
            results_text += f"   URL: {result.get('url', '')}\n"
            results_text += f"   Content: {result.get('content', result.get('snippet', ''))[:500]}...\n"
            results_text += f"   Source: {result.get('source', 'unknown')}\n"
        
        summary_prompt = f"""
        Summarize these web search results for the user request:
        
        User Request: {user_request}
        
        Web Results:
        {results_text}
        
        Provide a clear, comprehensive summary that directly answers the user's request.
        Include relevant URLs if the user might want to visit them.
        If there's an AI-generated answer, incorporate it appropriately.
        Make the response conversational and helpful.
        """
        
        try:
            message = await asyncio.to_thread(
                self.anthropic.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            # Fallback summary
            if results:
                fallback = f"Found {len(results)} results for '{user_request}':\n\n"
                for result in results[:3]:
                    fallback += f"• {result.get('title', 'Unknown')}\n"
                    fallback += f"  {result.get('snippet', 'No description available')}\n"
                    if result.get('url'):
                        fallback += f"  URL: {result['url']}\n"
                    fallback += "\n"
                return fallback
            else:
                return f"I couldn't find specific results for '{user_request}', but you can try searching directly on the web."
    
    async def get_current_news(self, topic: str = "", max_results: int = 5) -> List[Dict[str, Any]]:
        """Get current news using Tavily's news search"""
        
        try:
            query = f"latest news {topic}" if topic else "latest news"
            
            news_response = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                search_depth="advanced",
                include_answer=True,
                max_results=max_results,
                days=1  # Recent news only
            )
            
            return news_response.get('results', [])
            
        except Exception as e:
            raise Exception(f"News search failed: {str(e)}")
    
    def get_search_usage(self) -> Dict[str, Any]:
        """Get Tavily API usage information"""
        try:
            # Note: Tavily doesn't have a direct usage endpoint in the current API
            # This is a placeholder for future implementation
            return {
                "status": "active",
                "message": "Tavily search is operational"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
