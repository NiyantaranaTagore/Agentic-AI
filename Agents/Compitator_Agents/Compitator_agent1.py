import streamlit as st
import pandas as pd
from exa_py import Exa
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.firecrawl import FirecrawlTools
from phi.tools.duckduckgo import DuckDuckGo


# --- Streamlit Configuration ---
st.set_page_config(page_title="AI Competitor Intelligence Agent Team", layout="wide")

# --- Sidebar API Key Input ---
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")
exa_api_key = st.sidebar.text_input("Exa API Key", type="password")

# --- API Key Validation ---
if openai_api_key and firecrawl_api_key and exa_api_key:
    st.session_state.openai_api_key = openai_api_key
    st.session_state.firecrawl_api_key = firecrawl_api_key
    st.session_state.exa_api_key = exa_api_key
else:
    st.sidebar.warning("Please enter all API keys to proceed.")

# --- Inputs ---
url = st.text_input("Enter your company URL:")
description = st.text_area("Enter a description of your company (if URL is not available):")


# --- Agent Creation Utility ---
def create_agent(api_key, tools=None):
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=api_key),
        tools=tools or [],
        show_tool_calls=True,
        markdown=True
    )


# --- Get Similar Companies ---
def get_competitor_urls(url=None, description=None):
    exa = Exa(api_key=st.session_state.exa_api_key)
    if url:
        result = exa.find_similar(url=url, num_results=3, exclude_source_domain=True, category="company")
    elif description:
        result = exa.search(description, type="neural", category="company", use_autoprompt=True, num_results=3)
    else:
        raise ValueError("Please provide either a URL or a description.")

    return [item.url for item in result.results]


# --- Extract Data from Each Competitor Website ---
def extract_competitor_info(agent, competitor_url: str):
    try:
        response = agent.run(f"Crawl and summarize {competitor_url}")
        if not response or not response.content:
            raise ValueError("Empty response from agent.")
        return {"competitor": competitor_url, "data": response.content}
    except Exception as e:
        st.error(f"Error extracting info for {competitor_url}: {e}")
        return {"competitor": competitor_url, "error": str(e)}


# --- Generate Structured Markdown Table from Competitor Data ---
def generate_comparison_report(agent, competitor_data: list):
    combined_data = "\n\n".join([str(data) for data in competitor_data])
    system_prompt = """
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
    3. Keep entries concise but informative
    4. Use pipe symbols (|) to separate columns
    5. Include the separator row (|---|) after headers

    Competitor Data:
    {combined_data}
    """
    response = agent.run(system_prompt.format(combined_data=combined_data))

    st.subheader("Competitor Comparison")
    st.markdown(response.content)

    # Try to parse the markdown table into a DataFrame
    try:
        table_lines = [line.strip() for line in response.content.split('\n') if line.strip() and '|' in line]
        headers = [col.strip() for col in table_lines[0].split('|') if col.strip()]
        data_rows = [
            [cell.strip() for cell in line.split('|') if cell.strip()]
            for line in table_lines[2:]
        ]
        df = pd.DataFrame(data_rows, columns=headers)
        # Optional: Uncomment to show as DataFrame
        # st.table(df)
    except Exception as e:
        st.error(f"Error converting markdown to table: {e}")
        st.write("Raw markdown for debugging:", table_lines)


# --- Generate Strategic Analysis Based on Competitor Data ---
def generate_analysis_report(agent, competitor_data: list):
    combined_data = "\n\n".join([str(data) for data in competitor_data])
    prompt = f"""
    Analyze the following competitor data and identify market opportunities to improve my own company:
    {combined_data}

    Tasks:
    1. Identify market gaps and opportunities based on competitor offerings
    2. Analyze competitor weaknesses that we can capitalize on
    3. Recommend unique features or capabilities we should develop
    4. Suggest pricing and positioning strategies to gain competitive advantage
    5. Outline specific growth opportunities in underserved market segments
    6. Provide actionable recommendations for product development and go-to-market strategy

    Focus on finding opportunities where we can differentiate and do better than competitors.
    Highlight any unmet customer needs or pain points we can address.
    """
    return agent.run(prompt).content


# --- Main Workflow Function ---
def run_competitor_analysis(url, description):
    # Initialize tools and agents
    firecrawl_tools = FirecrawlTools(api_key=st.session_state.firecrawl_api_key, scrape=False, crawl=True, limit=5)
    firecrawl_agent = create_agent(st.session_state.openai_api_key, [firecrawl_tools, DuckDuckGo()])
    analysis_agent = create_agent(st.session_state.openai_api_key)
    comparison_agent = create_agent(st.session_state.openai_api_key)

    # Get competitor URLs
    with st.spinner("Fetching competitor URLs..."):
        competitor_urls = get_competitor_urls(url=url, description=description)
        st.write(f"Competitor URLs: {competitor_urls}")

    # Crawl competitor sites
    competitor_data = []
    for comp_url in competitor_urls:
        with st.spinner(f"Analyzing {comp_url}..."):
            data = extract_competitor_info(firecrawl_agent, comp_url)
            competitor_data.append(data)

    # Generate comparison table
    with st.spinner("Generating comparison table..."):
        generate_comparison_report(comparison_agent, competitor_data)

    # Generate strategic analysis
    with st.spinner("Generating analysis report..."):
        analysis = generate_analysis_report(analysis_agent, competitor_data)
        st.subheader("Competitor Analysis Report")
        st.markdown(analysis)

    st.success("Analysis complete!")


# --- Button Trigger ---
if st.button("Analyze Competitors"):
    if url or description:
        run_competitor_analysis(url, description)
    else:
        st.error("Please provide either a URL or a description.")
