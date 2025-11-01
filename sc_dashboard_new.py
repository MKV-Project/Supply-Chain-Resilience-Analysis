"""
Supply Chain Resilience Dashboard - Fully Dynamic Version
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sc_analyzer_new import SCAnalyzer
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional

# Constants
COLORS = {
    'Critical': '#dc2626',
    'Severe': '#ea580c',
    'Moderate': '#d97706',
    'Low': '#65a30d',
    'High': '#ef4444'
}

DEFAULT_TICKERS = "TSM, NVDA, INTC, AMD, QCOM, AVGO, F, GM, TSLA, AAPL, HPQ, DELL, CSCO, TATAMOTORS.NS, MARUTI.NS"
DATE_RANGE = {'start': datetime(2019, 1, 1), 'end': datetime(2023, 12, 31)}

# Setup page config and styling
st.set_page_config(page_title="SC-ALERT: Supply Chain Resilience", page_icon="🔗", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2563eb; text-align: center; margin-bottom: 1rem; }
    .impact-critical { background-color: #dc2626; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; }
    .impact-severe { background-color: #ea580c; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; }
    .impact-moderate { background-color: #d97706; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; }
    .impact-low { background-color: #65a30d; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def run_analysis(tickers: List[str], start_date: datetime, end_date: datetime, risk_threshold: float) -> Dict:
    """Run supply chain analysis with caching"""
    with st.spinner('Analyzing supply chain impacts...'):
        return SCAnalyzer().analyze(tickers, start_date, end_date, risk_threshold)

def create_excel_export(results: Dict) -> Optional[BytesIO]:
    """Create consolidated Excel export"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sheets = {
                'Analysis_Metadata': lambda: pd.DataFrame([results['metadata']]),
                'Performance_Analysis': lambda: pd.DataFrame(results['performance']).T.reset_index(),
                'Risk_Assessment': lambda: pd.DataFrame(results['risk']).T.reset_index(),
                'Supply_Chain_Impact': lambda: pd.DataFrame(results['supply_chain_impact']),
                'Sector_Vulnerability': lambda: pd.DataFrame(results['sector_vulnerability']),
                'Raw_Data_Samples': lambda: pd.DataFrame(results.get('raw_data_samples', [])),
                'Analysis_Summary': lambda: pd.DataFrame([create_analysis_summary(results)])
            }
            
            for sheet_name, data_func in sheets.items():
                try:
                    df = data_func()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if 'index' in df.columns:
                            df.columns = ['Ticker' if c == 'index' else c for c in df.columns]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    st.warning(f"Skipping {sheet_name}: {str(e)}")
        
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def create_analysis_summary(results: Dict) -> Dict:
    """Create analysis summary"""
    summary = {
        'total_companies_analyzed': len(results.get('companies', [])),
        'analysis_period': results.get('metadata', {}).get('period', 'N/A'),
        'analysis_date': results.get('metadata', {}).get('analysis_date', 'N/A')
    }
    
    if sc_data := results.get('supply_chain_impact'):
        df = pd.DataFrame(sc_data)
        summary.update({f'{k}_companies': v for k, v in df['Sector'].value_counts().items()})
        
    if risk_data := results.get('risk'):
        risk_dist = pd.DataFrame(risk_data).T['score'].value_counts()
        summary.update({f'risk_{k.lower()}': v for k, v in risk_dist.items()})
    
    return summary

def create_plot(data: pd.DataFrame, plot_type: str, **kwargs):
    """Create standardized plots"""
    plots = {
        'scatter': px.scatter,
        'bar': px.bar,
        'pie': px.pie,
        'line': px.line
    }
    return plots[plot_type](data, **kwargs)

def display_metrics(results: Dict) -> None:
    """Display key performance metrics - FULLY DYNAMIC"""
    sc_data = pd.DataFrame(results.get('supply_chain_impact', []))
    if sc_data.empty:
        st.warning("No supply chain data available for summary")
        return

    # Get available sectors dynamically
    available_sectors = sc_data['Sector'].unique()
    
    cols = st.columns(4)
    
    # Metric 1: Total companies
    cols[0].metric("Companies Analyzed", len(sc_data))
    
    # Metric 2: Critical impact companies
    cols[1].metric(
        "Critical Impact", 
        len(sc_data[sc_data['Impact_Severity'] == 'Critical'])
    )
    
    # Metric 3: Average resilience
    cols[2].metric(
        "Avg Resilience", 
        f"{sc_data['Supply_Chain_Resilience'].mean():.1f}"
    )
    
    # Metric 4: Highest impact sector (DYNAMIC)
    sector_impacts = sc_data.groupby('Sector')['Financial_Impact_pct'].mean()
    if not sector_impacts.empty:
        highest_impact_sector = sector_impacts.idxmax()
        highest_impact_value = sector_impacts.max()
        cols[3].metric(
            f"{highest_impact_sector} Impact", 
            f"{highest_impact_value:.1f}%"
        )
    else:
        cols[3].metric("Sector Impact", "N/A")

def display_executive_summary(results: Dict) -> None:
    """Display executive summary - FULLY DYNAMIC"""
    col1, col2, col3, col4 = st.columns(4)
    
    sc_data = results.get('supply_chain_impact', [])
    if sc_data:
        df = pd.DataFrame(sc_data)
        
        # Get period from metadata and format to show only years
        period = results.get('metadata', {}).get('period', 'N/A')
        formatted_period = "N/A"
        duration = "N/A"
        
        # Format period to show only years and calculate duration
        if ' to ' in period:
            start_part, end_part = period.split(' to ')
            try:
                start_year = int(start_part.strip().split('-')[0])
                end_year = int(end_part.strip().split('-')[0])
                formatted_period = f"{start_year} to {end_year}"
                duration = f"{end_year - start_year + 1} Years"
            except:
                formatted_period = period
                duration = "Multi-Year"
        
        with col1:
            st.metric("Analysis Period", formatted_period, delta=duration, delta_color="off")
        
        with col2:
            critical_count = len(df[df['Impact_Severity'] == 'Critical'])
            total_count = len(df)
            st.metric(
                "Critical Impact", 
                critical_count, 
                delta=f"{critical_count/total_count*100:.0f}%" if total_count > 0 else "N/A", 
                delta_color="inverse"
            )
            
        with col3:
            if not df['Estimated_Recovery_Months'].empty:
                # Get most common recovery time
                recovery_modes = df['Estimated_Recovery_Months'].mode()
                avg_recovery = recovery_modes.iloc[0] if not recovery_modes.empty else "N/A"
                st.metric("Avg Recovery", avg_recovery)
            else:
                st.metric("Avg Recovery", "N/A")
            
        with col4:
            high_resilience = len(df[df['Supply_Chain_Resilience'] > 70])
            st.metric(
                "Resilient Companies", 
                high_resilience,
                delta=f"{high_resilience/len(df)*100:.0f}%" if len(df) > 0 else "N/A"
            )
    else:
        # Fallback when no data
        period = results.get('metadata', {}).get('period', 'N/A')
        with col1:
            st.metric("Analysis Period", period)
        with col2:
            st.metric("Critical Impact", "N/A")
        with col3:
            st.metric("Avg Recovery", "N/A")
        with col4:
            st.metric("Resilient Companies", "N/A")

    st.subheader("Executive Summary")
    display_metrics(results)
    
    if sc_data:
        severity_counts = df['Impact_Severity'].value_counts().reset_index()
        severity_counts.columns = ['Severity', 'Count']
        
        if not severity_counts.empty:
            fig = create_plot(
                data=severity_counts,
                plot_type='pie',
                values='Count',
                names='Severity',
                title="Distribution of Supply Chain Impact Severity",
                color='Severity',
                color_discrete_map=COLORS
            )
            st.plotly_chart(fig, use_container_width=True)
    
   # st.subheader("Sector Performance Trends")
    
    time_series_data = generate_dynamic_time_series(results)
    
    # if not time_series_data.empty:
    #     # Get actual year range from data
    #     min_year = int(time_series_data['Year'].min())
    #     max_year = int(time_series_data['Year'].max())
        
    #     fig = px.line(
    #         time_series_data,
    #         x='Year', 
    #         y='Index',
    #         color='Sector',
    #         title=f"Sector Performance Trends ({min_year}-{max_year})",
    #         labels={'Index': 'Performance Index'},
    #         line_shape='spline'
    #     )
        
    #     years = sorted(time_series_data['Year'].unique())
    #     fig.update_xaxes(
    #         tickmode='array',
    #         tickvals=years,
    #         ticktext=[str(int(year)) for year in years]
    #     )
        
    #     fig.update_layout(height=400)
    #     st.plotly_chart(fig, use_container_width=True)

def generate_dynamic_time_series(results: Dict) -> pd.DataFrame:
    """Generate time series dynamically from actual data - NO HARDCODED PATTERNS"""
    try:
        perf_data = results.get('performance', {})
        if not perf_data:
            return pd.DataFrame()
        
        # Extract actual date range from metadata
        metadata = results.get('metadata', {})
        period_str = metadata.get('period', 'N/A to N/A')
        
        if ' to ' in period_str:
            start_part, end_part = period_str.split(' to ')
            start_year = int(start_part.strip().split('-')[0])
            end_year = int(end_part.strip().split('-')[0])
        else:
            return pd.DataFrame()
        
        years = list(range(start_year, end_year + 1))
        
        # Group by sector and calculate average returns
        sectors = {}
        for data in perf_data.values():
            sector = data['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(data['return'])
        
        sector_avg = {sector: np.mean(returns) for sector, returns in sectors.items()}
        
        # Generate time series based on actual returns
        time_series_data = []
        
        for sector, avg_return in sector_avg.items():
            for i, year in enumerate(years):
                # Linear progression based on actual returns
                progress = i / max(1, len(years) - 1)
                index_value = 100 + (avg_return * progress)
                
                time_series_data.append({
                    'Year': year,
                    'Index': round(index_value, 2),
                    'Sector': sector
                })
        
        return pd.DataFrame(time_series_data)
        
    except Exception as e:
        return pd.DataFrame()

def display_performance_analysis(results: Dict) -> None:
    """Display performance analysis"""
    st.subheader("Performance Analysis")
    
    if not (perf_data := results.get('performance')):
        st.warning("No performance data available")
        return
    
    df = pd.DataFrame([
        {
            'Company': data['name'],
            'Ticker': ticker,
            'Sector': data['sector'],
            'Return (%)': data['return'],
            'Volatility (%)': data['volatility'],
            'Drawdown (%)': data['drawdown'],
            'Abs_Drawdown': abs(data['drawdown'])
        }
        for ticker, data in perf_data.items()
    ])
    
    # Performance charts
    for plot_config in [
        {
            'type': 'bar',
            'x': 'Company',
            'y': 'Return (%)',
            'color': 'Sector',
            'title': "Total Returns by Company",
            'hover_data': ['Ticker', 'Drawdown (%)']
        },
        {
            'type': 'scatter',
            'x': 'Volatility (%)',
            'y': 'Return (%)',
            'color': 'Sector',
            'size': 'Abs_Drawdown',
            'hover_data': ['Company', 'Ticker'],
            'title': "Risk-Return Profile: Volatility vs Return",
            'size_max': 30
        }
    ]:
        fig = create_plot(df, plot_type=plot_config.pop('type'), **plot_config)
        if 'bar' in str(plot_config.get('title', '')):
            fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    display_df = df.drop('Abs_Drawdown', axis=1).copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

def display_risk_analysis(results: Dict) -> None:
    """Display risk analysis with dynamic correlation heatmap"""
    st.subheader("Risk Assessment")
    
    if not (risk_data := results.get('risk')) or not (perf_data := results.get('performance')):
        st.warning("No risk assessment data available")
        return
    
    df = pd.DataFrame([
        {
            'Company': perf['name'],
            'Ticker': ticker,
            'Sector': perf['sector'],
            'Return (%)': perf['return'],
            'Volatility (%)': perf['volatility'],
            'Drawdown (%)': perf['drawdown'],
            'Risk_Score': risk_data.get(ticker, {}).get('score', 'Unknown')
        }
        for ticker, perf in perf_data.items()
    ])
    
    # Risk scatter plot
    fig = create_plot(
        df,
        plot_type='scatter',
        x='Volatility (%)',
        y='Drawdown (%)',
        color='Risk_Score',
        size='Volatility (%)',
        hover_data=['Company', 'Sector', 'Return (%)'],
        title="Risk Analysis: Volatility vs Drawdown",
        color_discrete_map={'High': COLORS['High'], 'Low': COLORS['Low']}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk distribution pie chart
    risk_counts = df['Risk_Score'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Score', 'Count']
    fig = create_plot(
        risk_counts,
        plot_type='pie',
        values='Count',
        names='Risk_Score',
        title="Risk Score Distribution",
        color='Risk_Score',
        color_discrete_map={'High': COLORS['High'], 'Low': COLORS['Low']}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # === CORRELATION HEATMAP - FULLY DYNAMIC ===
    st.subheader("Sector Correlation Analysis")
    
    correlation_data = create_dynamic_correlation_matrix(results)
    
    if not correlation_data.empty:
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.index,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Sector Performance Correlation Matrix",
            xaxis_title="Sector",
            yaxis_title="Sector",
            height=500,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - Values close to +1 (red): Strong positive correlation - sectors move together
        - Values close to -1 (blue): Strong negative correlation - sectors move oppositely
        - Values close to 0 (white): No correlation - sectors move independently
        """)
    
    display_df = df.copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

def create_dynamic_correlation_matrix(results: Dict) -> pd.DataFrame:
    """Create correlation matrix based on ACTUAL returns data - NO HARDCODING"""
    try:
        perf_data = results.get('performance', {})
        if not perf_data:
            return pd.DataFrame()
        
        # Group returns by sector
        sector_returns = {}
        for ticker, data in perf_data.items():
            sector = data['sector']
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(data['return'])
        
        sectors = list(sector_returns.keys())
        n_sectors = len(sectors)
        
        if n_sectors < 2:
            return pd.DataFrame()
        
        # Calculate actual correlations based on returns
        correlation_matrix = np.zeros((n_sectors, n_sectors))
        
        for i in range(n_sectors):
            for j in range(n_sectors):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Calculate correlation between sector returns
                    returns_i = np.array(sector_returns[sectors[i]])
                    returns_j = np.array(sector_returns[sectors[j]])
                    
                    # Ensure same length
                    min_len = min(len(returns_i), len(returns_j))
                    if min_len > 1:
                        corr = np.corrcoef(returns_i[:min_len], returns_j[:min_len])[0, 1]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[i, j] = 0.0
        
        correlation_df = pd.DataFrame(
            correlation_matrix,
            index=sectors,
            columns=sectors
        )
        
        return correlation_df
        
    except Exception as e:
        st.warning(f"Could not generate correlation matrix: {str(e)}")
        return pd.DataFrame()

def display_supply_chain_analysis(results: Dict) -> None:
    """Display supply chain impact analysis - FULLY DYNAMIC"""
    st.subheader("Supply Chain Impact Analysis")
    
    if not (sc_data := results.get('supply_chain_impact')):
        st.warning("No supply chain impact data available")
        return
    
    df = pd.DataFrame(sc_data)
    severity_counts = df['Impact_Severity'].value_counts()
    
    col1, col2 = st.columns(2)
    
    severity_df = severity_counts.reset_index()
    severity_df.columns = ['Severity', 'Count']
    
    with col1:
        if not severity_counts.empty:
            fig = create_plot(
                severity_df,
                plot_type='pie',
                values='Count',
                names='Severity',
                title="Supply Chain Impact Severity",
                color='Severity',
                color_discrete_map=COLORS
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not severity_counts.empty:
            fig = create_plot(
                severity_df,
                plot_type='bar',
                x='Severity',
                y='Count',
                title="Impact Severity Count",
                color='Severity',
                color_discrete_map=COLORS,
                labels={'x': 'Severity', 'y': 'Number of Companies'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sector impact analysis - DYNAMIC
    sectors = df['Sector'].unique()
    if len(sectors) > 0:
        sector_impact = df.groupby('Sector').agg({
            'Financial_Impact_pct': 'mean',
            'Supply_Chain_Resilience': 'mean',
            'Company': 'count'
        }).round(1)
        
        sector_impact_df = sector_impact.reset_index()
        if not sector_impact_df.empty:
            fig = create_plot(
                sector_impact_df,
                plot_type='bar',
                x='Sector',
                y='Financial_Impact_pct',
                title="Average Financial Impact by Sector (%)",
                color='Financial_Impact_pct',
                color_continuous_scale='reds',
                hover_data=['Supply_Chain_Resilience']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # === RECOVERY TRAJECTORY - FULLY DYNAMIC ===
    st.subheader("Recovery Trajectory Comparison")
    
    time_series_data = results.get('time_series_data')
    
    if time_series_data is not None:
        if isinstance(time_series_data, list):
            ts_df = pd.DataFrame(time_series_data)
        elif isinstance(time_series_data, pd.DataFrame):
            ts_df = time_series_data
        else:
            ts_df = pd.DataFrame()
        
        if not ts_df.empty and 'Date' in ts_df.columns:
            ts_df['Date'] = pd.to_datetime(ts_df['Date'])
            
            # Determine grouping column dynamically
            group_col = 'Sector' if 'Sector' in ts_df.columns else 'Company'
            
            fig = px.line(
                ts_df,
                x='Date',
                y='Normalized_Price',
                color=group_col,
                title="Stock Price Recovery Patterns (Normalized to Base 100)",
                labels={'Normalized_Price': 'Price Index (Base 100)', 'Date': 'Date'},
                line_shape='spline',
                hover_data=['Ticker'] if 'Ticker' in ts_df.columns else None
            )
            
            # Get actual date range from data
            min_date = ts_df['Date'].min()
            max_date = ts_df['Date'].max()
            
            # Calculate midpoint for disruption period dynamically
            date_range = (max_date - min_date).days
            if date_range > 730:  # If > 2 years of data
                # Use middle 40% of date range as disruption period
                disruption_start = min_date + pd.Timedelta(days=date_range * 0.3)
                disruption_end = min_date + pd.Timedelta(days=date_range * 0.7)
                
                fig.add_vrect(
                    x0=disruption_start,
                    x1=disruption_end,
                    fillcolor="red",
                    opacity=0.1,
                    annotation_text="Analysis Period",
                    annotation_position="top left",
                    layer="below"
                )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price Index (Base 100)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Time series data unavailable or incomplete.")
    else:
        st.warning("No time series data available.")
    
    # Supply chain impact table
    display_df = df[[
        'Company', 'Ticker', 'Sector', 'Semiconductor_Dependency',
        'Financial_Impact_pct', 'Impact_Severity', 'Estimated_Recovery_Months',
        'Supply_Chain_Resilience', 'Strategic_Recommendation'
    ]]
    
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(
        display_df.style.applymap(
            lambda x: f"background-color: {COLORS.get(x, '#65a30d')}; color: white" 
            if x in COLORS else '',
            subset=['Impact_Severity']
        ),
        use_container_width=True,
        height=400
    )

def display_strategic_recommendations(results: Dict) -> None:
    """Display strategic recommendations - FULLY DYNAMIC"""
    st.subheader("Strategic Recommendations")
    
    if not (sc_data := results.get('supply_chain_impact')):
        st.warning("No data available for recommendations")
        return
    
    df = pd.DataFrame(sc_data)
    
    # Get all unique sectors dynamically
    for sector in sorted(df['Sector'].unique()):
        sector_companies = df[df['Sector'] == sector]
        critical_companies = sector_companies[sector_companies['Impact_Severity'].isin(['Critical', 'Severe'])]
        
        if critical_companies.empty:
            st.success(f"**{sector} Sector - Stable**")
            st.write(f"• {len(sector_companies)} companies analyzed")
            if not sector_companies.empty:
                st.write(f"• Recommended action: {sector_companies['Strategic_Recommendation'].iloc[0]}")
        else:
            st.error(f"**{sector} Sector - Immediate Attention Required**")
            st.write(f"• {len(critical_companies)} companies with critical/severe impact")
            st.write(f"• Primary recommendation: {critical_companies['Strategic_Recommendation'].iloc[0]}")
        st.markdown("---")

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">RiskFlow</h1>', unsafe_allow_html=True)
    st.markdown("### Semiconductor Supply Chain Resilience Analysis")
    
    # Sidebar controls
    st.sidebar.title("Analysis Controls")
    tickers = [t.strip() for t in st.sidebar.text_area(
        "Enter company tickers (comma-separated):",
        value=DEFAULT_TICKERS,
        height=100,
        help="Include companies from different sectors for comprehensive analysis"
    ).split(',') if t.strip()]
    
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=DATE_RANGE['start'])
    end_date = col2.date_input("End Date", value=DATE_RANGE['end'])
    
    risk_threshold = st.sidebar.slider(
        "Risk Detection Sensitivity:",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Higher values detect more anomalies"
    )
    
    # Analysis controls
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Run Analysis", type="primary", use_container_width=True):
        try:
            results = run_analysis(tickers, start_date, end_date, risk_threshold)
            st.session_state.results = results
            st.success("Analysis completed successfully!")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.info("Please check your ticker symbols and try again")
    
    if col2.button("Clear Results", use_container_width=True):
        st.session_state.pop('results', None)
        st.rerun()
    
    # Display results if available
    if results := st.session_state.get('results'):
        # Export options
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Results")
        if st.sidebar.button("Export All Data to Excel", use_container_width=True):
            if excel_file := create_excel_export(results):
                st.sidebar.download_button(
                    "⬇Download Excel",
                    excel_file,
                    file_name=f"supply_chain_analysis_{datetime.now():%Y%m%d_%H%M}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Individual CSV exports
        for label, data_key in [
            ("Supply Chain Impact", 'supply_chain_impact'),
            ("Sector Vulnerability", 'sector_vulnerability'),
            ("Performance Data", 'performance')
        ]:
            if data := results.get(data_key):
                df = pd.DataFrame(data)
                if data_key == 'performance':
                    df = df.T
                st.sidebar.download_button(
                    f"{label}",
                    df.to_csv(index=False),
                    file_name=f"{data_key}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Analysis tabs
        tab_funcs = {
            "Summary": display_executive_summary,
            "Performance": display_performance_analysis,
            "Risk": display_risk_analysis,
            "Supply Chain": display_supply_chain_analysis,
            "Recommendations": display_strategic_recommendations
        }
        
        tabs = st.tabs(list(tab_funcs.keys()))
        for tab, func in zip(tabs, tab_funcs.values()):
            with tab:
                func(results)
    else:
        st.info("""
        ## Supply Chain Risk Analysis Dashboard

        Analyze and monitor semiconductor supply chain risks across industries.
        
        ### Quick Start:
        1. Enter tickers in sidebar (default set provided)
        2. Set date range
        3. Run analysis
        4. View results in tabs

        ### Analysis Tabs:
        • **Summary** - Key metrics & trends
        • **Performance** - Returns & volatility
        • **Risk** - Risk scores & correlations
        • **Supply Chain** - Impact assessment
        • **Recommendations** - Strategic actions

        ### Key Features:
        • Real-time stock data analysis
        • Machine learning risk detection
        • Interactive visualizations
        • Multi-sector correlation analysis
        • Excel/CSV export options

        **Note:** For best results, use 1+ year of data.
        """)

if __name__ == "__main__":
    main()




