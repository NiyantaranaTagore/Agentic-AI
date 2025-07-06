"""
AI Competitor Intelligence Agent Team - Refined Version

This module provides a Streamlit application for competitive intelligence analysis
using AI agents to crawl, analyze, and compare competitor websites.
"""

import streamlit as st
import pandas as pd
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
import time

# Third-party imports
from exa_py import Exa
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.firecrawl import FirecrawlTools
from phi.tools.duckduckgo import DuckDuckGo


# Configuration and Constants
class Config:
    """Application configuration constants."""
    PAGE_TITLE = "AI Competitor Intelligence Agent Team"
    PAGE_LAYOUT = "wide"
    DEFAULT_MODEL = "gpt-4o-mini"
    MAX_COMPETITORS = 3
    CRAWL_LIMIT = 5
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0


@dataclass
class CompetitorData:
    """Data structure for competitor information."""
    url: str
    data: Optional[str] = None
    error: Optional[str] = None
    success: bool = False


class APIKeyManager:
    """Manages API key validation and storage."""
    
    REQUIRED_KEYS = ['openai_api_key', 'firecrawl_api_key', 'exa_api_key']
    
    @staticmethod
    def validate_and_store_keys() -> bool:
        """Validate and store API keys in session state."""
        api_keys = {
            'openai_api_key': st.sidebar.text_input("OpenAI API Key", type="password"),
            'firecrawl_api_key': st.sidebar.text_input("Firecrawl API Key", type="password"),
            'exa_api_key': st.sidebar.text_input("Exa API Key", type="password")
        }
        
        if all(api_keys.values()):
            for key, value in api_keys.items():
                st.session_state[key] = value
            return True
        else:
            st.sidebar.warning("Please enter all API keys to proceed.")
            return False
    
    @staticmethod
    def get_api_key(key_name: str) -> str:
        """Safely retrieve API key from session state."""
        return st.session_state.get(key_name, "")


class AgentFactory:
    """Factory class for creating AI agents with consistent configuration."""
    
    @staticmethod
    def create_agent(api_key: str, tools: Optional[List] = None) -> Agent:
        """Create an AI agent with specified tools."""
        if not api_key:
            raise ValueError("API key is required to create an agent")
        
        return Agent(
            model=OpenAIChat(id=Config.DEFAULT_MODEL, api_key=api_key),
            tools=tools or [],
            show_tool_calls=True,
            markdown=True
        )
    
    @staticmethod
    def create_firecrawl_agent(openai_key: str, firecrawl_key: str) -> Agent:
        """Create an agent specifically configured for web crawling."""
        firecrawl_tools = FirecrawlTools(
            api_key=firecrawl_key,
            scrape=False,
            crawl=True,
            limit=Config.CRAWL_LIMIT
        )
        return AgentFactory.create_agent(openai_key, [firecrawl_tools, DuckDuckGo()])


class CompetitorDiscovery:
    """Handles competitor discovery using Exa API."""
    
    def __init__(self, api_key: str):
        self.exa = Exa(api_key=api_key)
    
    def find_competitors(self, url: Optional[str] = None, 
                        description: Optional[str] = None) -> List[str]:
        """
        Find competitor URLs using either company URL or description.
        
        Args:
            url: Company URL for similarity search
            description: Company description for neural search
            
        Returns:
            List of competitor URLs
            
        Raises:
            ValueError: If neither URL nor description is provided
        """
        if not url and not description:
            raise ValueError("Please provide either a URL or a description.")
        
        try:
            if url:
                result = self.exa.find_similar(
                    url=url,
                    num_results=Config.MAX_COMPETITORS,
                    exclude_source_domain=True,
                    category="company"
                )
            else:
                result = self.exa.search(
                    description,
                    type="neural",
                    category="company",
                    use_autoprompt=True,
                    num_results=Config.MAX_COMPETITORS
                )
            
            return [item.url for item in result.results]
            
        except Exception as e:
            logging.error(f"Error finding competitors: {e}")
            raise


class CompetitorAnalyzer:
    """Handles competitor data extraction and analysis."""
    
    def __init__(self, crawl_agent: Agent, analysis_agent: Agent, comparison_agent: Agent):
        self.crawl_agent = crawl_agent
        self.analysis_agent = analysis_agent
        self.comparison_agent = comparison_agent
    
    @staticmethod
    def _retry_on_failure(max_attempts: int = Config.RETRY_ATTEMPTS):
        """Decorator for retrying failed operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise e
                        time.sleep(Config.RETRY_DELAY * (attempt + 1))
                        logging.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
            return wrapper
        return decorator
    
    @_retry_on_failure()
    def extract_competitor_info(self, competitor_url: str) -> CompetitorData:
        """
        Extract information from a competitor website.
        
        Args:
            competitor_url: URL of the competitor website
            
        Returns:
            CompetitorData object with extracted information or error details
        """
        try:
            response = self.crawl_agent.run(f"Crawl and summarize {competitor_url}")
            
            if not response or not response.content:
                raise ValueError("Empty response from crawling agent")
            
            return CompetitorData(
                url=competitor_url,
                data=response.content,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error extracting info for {competitor_url}: {str(e)}"
            logging.error(error_msg)
            return CompetitorData(
                url=competitor_url,
                error=error_msg,
                success=False
            )
    
    def generate_comparison_table(self, competitor_data: List[CompetitorData]) -> None:
        """Generate and display a structured comparison table."""
        successful_data = [data for data in competitor_data if data.success and data.data]
        
        if not successful_data:
            st.error("No competitor data available for comparison.")
            return
        
        combined_data = "\n\n".join([data.data for data in successful_data])
        
        system_prompt = self._get_comparison_prompt_template()
        
        try:
            response = self.comparison_agent.run(
                system_prompt.format(combined_data=combined_data)
            )
            
            st.subheader("📊 Competitor Comparison")
            st.markdown(response.content)
            
            # Attempt to create downloadable DataFrame
            self._create_downloadable_table(response.content)
            
        except Exception as e:
            st.error(f"Error generating comparison table: {e}")
            logging.error(f"Comparison table generation failed: {e}")
    
    def generate_strategic_analysis(self, competitor_data: List[CompetitorData]) -> str:
        """Generate strategic analysis and recommendations."""
        successful_data = [data for data in competitor_data if data.success and data.data]
        
        if not successful_data:
            return "No competitor data available for strategic analysis."
        
        combined_data = "\n\n".join([data.data for data in successful_data])
        
        prompt = self._get_analysis_prompt_template().format(combined_data=combined_data)
        
        try:
            response = self.analysis_agent.run(prompt)
            return response.content
        except Exception as e:
            error_msg = f"Error generating strategic analysis: {e}"
            logging.error(error_msg)
            return error_msg
    
    @staticmethod
    def _get_comparison_prompt_template() -> str:
        """Get the prompt template for comparison table generation."""
        return """
        As an expert business analyst, analyze the competitor data and create a structured comparison table.
        Format the data in EXACTLY this markdown table structure:
        
        | Company | Pricing | Key Features | Tech Stack | Marketing Focus | Customer Feedback |
        |---------|---------|--------------|------------|-----------------|-------------------|
        | [Company Name 1] | ... | ... | ... | ... | ... |
        | [Company Name 2] | ... | ... | ... | ... | ... |
        | [Company Name 3] | ... | ... | ... | ... | ... |
        
        Rules:
        1. Always include all columns
        2. Use the exact column names specified above
        3. Keep entries concise but informative (max 50 words per cell)
        4. Use pipe symbols (|) to separate columns
        5. Include the separator row (|---|) after headers
        6. If data is missing, use "N/A" or "Not Available"
        
        Competitor Data:
        {combined_data}
        """
    
    @staticmethod
    def _get_analysis_prompt_template() -> str:
        """Get the prompt template for strategic analysis."""
        return """
        Analyze the following competitor data and provide strategic insights:
        
        {combined_data}
        
        Please provide a comprehensive analysis covering:
        
        ## 🎯 Market Opportunities
        - Identify gaps in competitor offerings
        - Highlight underserved market segments
        
        ## ⚡ Competitive Advantages
        - Analyze competitor weaknesses we can exploit
        - Recommend differentiation strategies
        
        ## 💡 Product Development Recommendations
        - Suggest unique features or capabilities to develop
        - Identify technology or service gaps
        
        ## 💰 Pricing & Positioning Strategy
        - Recommend competitive pricing approaches
        - Suggest optimal market positioning
        
        ## 🚀 Growth Opportunities
        - Outline specific expansion opportunities
        - Identify untapped customer segments
        
        ## 📋 Actionable Next Steps
        - Provide concrete, implementable recommendations
        - Prioritize initiatives by potential impact
        
        Focus on actionable insights that can drive competitive advantage and business growth.
        """
    
    def _create_downloadable_table(self, markdown_content: str) -> None:
        """Create a downloadable DataFrame from markdown table."""
        try:
            table_lines = [
                line.strip() for line in markdown_content.split('\n')
                if line.strip() and '|' in line and not line.strip().startswith('|--')
            ]
            
            if len(table_lines) < 2:
                return
            
            headers = [col.strip() for col in table_lines[0].split('|') if col.strip()]
            data_rows = []
            
            for line in table_lines[1:]:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(row) == len(headers):
                    data_rows.append(row)
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                
                # Provide download option
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Table as CSV",
                    data=csv_data,
                    file_name="competitor_comparison.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            logging.warning(f"Could not create downloadable table: {e}")


@contextmanager
def loading_spinner(message: str):
    """Context manager for consistent loading spinners."""
    with st.spinner(message):
        yield


class CompetitorIntelligenceApp:
    """Main application class that orchestrates the competitor intelligence workflow."""
    
    def __init__(self):
        self.setup_logging()
        self.setup_page_config()
    
    @staticmethod
    def setup_logging():
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            layout=Config.PAGE_LAYOUT,
            page_icon="🔍"
        )
    
    def render_header(self):
        """Render the application header."""
        st.title("🔍 AI Competitor Intelligence Agent Team")
        st.markdown("""
        Discover and analyze your competitors using AI-powered web crawling and analysis.
        Get strategic insights to improve your competitive position.
        """)
    
    def render_inputs(self) -> tuple[str, str]:
        """Render input fields and return user inputs."""
        col1, col2 = st.columns(2)
        
        with col1:
            url = st.text_input(
                "🌐 Enter your company URL:",
                placeholder="https://yourcompany.com",
                help="We'll find similar companies to analyze"
            )
        
        with col2:
            description = st.text_area(
                "📝 Or describe your company:",
                placeholder="e.g., AI-powered project management tool for remote teams",
                help="Use this if you don't have a URL yet"
            )
        
        return url.strip(), description.strip()
    
    def run_analysis(self, url: str, description: str):
        """Execute the complete competitor analysis workflow."""
        try:
            # Initialize components
            discovery = CompetitorDiscovery(APIKeyManager.get_api_key('exa_api_key'))
            
            # Create agents
            crawl_agent = AgentFactory.create_firecrawl_agent(
                APIKeyManager.get_api_key('openai_api_key'),
                APIKeyManager.get_api_key('firecrawl_api_key')
            )
            analysis_agent = AgentFactory.create_agent(APIKeyManager.get_api_key('openai_api_key'))
            comparison_agent = AgentFactory.create_agent(APIKeyManager.get_api_key('openai_api_key'))
            
            analyzer = CompetitorAnalyzer(crawl_agent, analysis_agent, comparison_agent)
            
            # Step 1: Find competitors
            with loading_spinner("🔍 Discovering competitors..."):
                competitor_urls = discovery.find_competitors(url=url, description=description)
                
                if not competitor_urls:
                    st.warning("No competitors found. Try adjusting your search criteria.")
                    return
                
                st.success(f"Found {len(competitor_urls)} competitors!")
                
                # Display found competitors
                with st.expander("🏢 Discovered Competitors", expanded=True):
                    for i, comp_url in enumerate(competitor_urls, 1):
                        st.write(f"{i}. {comp_url}")
            
            # Step 2: Extract competitor data
            competitor_data = []
            progress_bar = st.progress(0)
            
            for i, comp_url in enumerate(competitor_urls):
                with loading_spinner(f"📊 Analyzing competitor {i+1}/{len(competitor_urls)}..."):
                    data = analyzer.extract_competitor_info(comp_url)
                    competitor_data.append(data)
                    progress_bar.progress((i + 1) / len(competitor_urls))
            
            # Display extraction results
            successful_extractions = sum(1 for data in competitor_data if data.success)
            if successful_extractions == 0:
                st.error("Failed to extract data from any competitor websites.")
                return
            
            st.info(f"Successfully analyzed {successful_extractions}/{len(competitor_urls)} competitors")
            
            # Step 3: Generate comparison table
            with loading_spinner("📊 Generating comparison table..."):
                analyzer.generate_comparison_table(competitor_data)
            
            # Step 4: Generate strategic analysis
            with loading_spinner("🧠 Generating strategic insights..."):
                analysis = analyzer.generate_strategic_analysis(competitor_data)
                
                st.subheader("🎯 Strategic Analysis & Recommendations")
                st.markdown(analysis)
            
            st.success("✅ Analysis complete! Use the insights above to strengthen your competitive position.")
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            logging.error(f"Analysis workflow failed: {e}")
    
    def run(self):
        """Main application entry point."""
        self.render_header()
        
        # API Key Management
        if not APIKeyManager.validate_and_store_keys():
            st.stop()
        
        # User Inputs
        url, description = self.render_inputs()
        
        # Validation and Analysis Trigger
        if st.button("🚀 Analyze Competitors", type="primary"):
            if not url and not description:
                st.error("Please provide either a company URL or description.")
                return
            
            self.run_analysis(url, description)


# Application Entry Point
def main():
    """Main function to run the Streamlit application."""
    app = CompetitorIntelligenceApp()
    app.run()


if __name__ == "__main__":
    main()