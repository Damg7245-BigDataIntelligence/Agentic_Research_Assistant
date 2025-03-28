from typing import Dict, List, Any
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
from datetime import datetime
from llm_service import generate_response_with_gemini  # Add this import

# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

class WebSearchAgent:
    def __init__(self):
        self.api_key = SERPAPI_API_KEY
        
    def search_news(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for recent news articles about NVIDIA
        """
        print(f"Searching for news articles about {query}")
        try:
            nvidia_query = f"NVIDIA {query}"
            search_params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": nvidia_query,
                "num": num_results,
                "tbm": "nws",  # News results
                "tbs": "qdr:m",  # Last month's results
                "location": "United States"
            }
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            formatted_results = []
            if "news_results" in results:
                for item in results["news_results"]:
                    formatted_results.append({
                        "type": "news",
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": item.get("source", ""),
                        "date": item.get("date", ""),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in news search: {str(e)}")
            return []

    def search_trends(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for general trends and articles about NVIDIA
        """
        try:
            nvidia_query = f"NVIDIA {query} trends analysis research"
            search_params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": nvidia_query,
                "num": num_results,
                "tbs": "qdr:m"  # Last month's results
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            formatted_results = []
            if "organic_results" in results:
                for item in results["organic_results"]:
                    formatted_results.append({
                        "type": "trend",
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": item.get("source", "website"),
                        "date": item.get("date", "Recent"),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in trends search: {str(e)}")
            return []

    def process_results(self, news_results: List[Dict[str, Any]], trend_results: List[Dict[str, Any]]) -> str:
        """
        Process and format both news and trend results into a comprehensive summary
        """
        if not news_results and not trend_results:
            return "No relevant results found."
            
        summary = "NVIDIA Market Intelligence Report:\n\n"
        
        # Process News Section
        if news_results:
            summary += "📰 Latest News:\n" + "="*50 + "\n"
            for i, result in enumerate(news_results, 1):
                summary += f"{i}. {result['title']}\n"
                summary += f"   📅 {result['date']} | 🔍 {result['source']}\n"
                summary += f"   {result['snippet']}\n"
                summary += f"   🔗 {result['link']}\n\n"
        
        # Process Trends Section
        if trend_results:
            summary += "\n📈 Market Trends & Analysis:\n" + "="*50 + "\n"
            for i, result in enumerate(trend_results, 1):
                summary += f"{i}. {result['title']}\n"
                summary += f"   💡 Key Points: {result['snippet']}\n"
                summary += f"   🔗 {result['link']}\n\n"
        
        # Add timestamp
        summary += f"\n🕒 Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return summary

    def synthesize_results(self, news_results: List[Dict], trend_results: List[Dict]) -> str:
        """
        Create an analytical summary using Gemini based on news and trend snippets
        """
        # Prepare context from news and trends
        news_context = "\n".join([
            f"NEWS ARTICLE:\n"
            f"Title: {item['title']}\n"
            f"Date: {item['date']}\n"
            f"Source: {item['source']}\n"
            f"Summary: {item['snippet']}\n"
            for item in news_results
        ])

        trends_context = "\n".join([
            f"MARKET TREND:\n"
            f"Title: {item['title']}\n"
            f"Summary: {item['snippet']}\n"
            for item in trend_results
        ])

        context = f"""
        RECENT NEWS AND TRENDS ABOUT NVIDIA:

        {news_context}

        MARKET TRENDS AND ANALYSIS:
        {trends_context}
        """

        print("web search context: ", context)
        # Use the new response_type parameter
        analysis, token_info = generate_response_with_gemini(
            query="Analyze NVIDIA updates",
            context=context,
            response_type="web_analysis"
        )
        print("web search analysis: ", analysis)
        return analysis, token_info  # Now also returning token info for tracking

    def run(self, query: str) -> Dict[str, Any]:
        """
        Modified run method to include synthesis
        """
        try:
            # Perform searches
            news_results = self.search_news(query)
            trend_results = self.search_trends(query)
            print("news_results: ", news_results)
            print("trend_results: ", trend_results)
            # Generate basic summary
            summary = self.process_results(news_results, trend_results)
            
            # Generate analytical insights
            insights, token_info = self.synthesize_results(news_results, trend_results)
            
            return {
                "status": "success",
                "summary": summary,
                "insights": insights,  # New synthesized analysis
                "token_info": token_info,
                "raw_results": {
                    "news": news_results,
                    "trends": trend_results
                },
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "categories": {
                    "has_news": bool(news_results),
                    "has_trends": bool(trend_results)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            } 