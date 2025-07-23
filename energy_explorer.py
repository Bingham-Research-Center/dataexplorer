import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#Google Analytics tracking
def add_google_analytics():
    """Add Google Analytics tracking to the app"""
    ga_code = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-SHRBGEW9GD"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-SHRBGEW9GD');
    </script>
    """
    components.html(ga_code, height=0)



# Set page config
st.set_page_config(
    page_title="Uinta Basin Emissions Data Explorer",
    page_icon="⚡",
    layout="wide"
    # Remove the initial_sidebar_state line
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stPlotlyChart {
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data
def load_data():
    """Load and prepare the dataset from the Excel file"""
    try:
        # Load the data from the Excel file
        df = pd.read_excel('mastertimeseries.xlsx', sheet_name='data')
        
        # Clean column names - replace spaces and special characters with underscores
        df.columns = df.columns.str.replace(' ', '_').str.replace(':', '_').str.replace('-', '_')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Fallback to sample data if file not found
        data = {
            'year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            'oil_million_bbls': [27.05, 32.95, 29.89, 24.01, 28.26, 31.15, 31.16, 26.07, 31.18, 35.77, 45.33],
            'gas_million_bbleq': [61.68, 59.88, 52.90, 47.00, 40.73, 38.41, 35.33, 31.28, 31.23, 40.01, 43.42],
            'energy_million_bbleq': [88.72, 92.83, 82.79, 71.01, 68.99, 69.56, 66.49, 57.35, 62.41, 75.79, 88.75]
        }
        return pd.DataFrame(data)


# Custom display names for variables
display_names = {
    'year': 'Year',
    'basinwide_ch4_emiss_Mg_hr': 'Uinta Basin-wide methane emissions (Mg/hr)',
    'oil_million_bbls': 'Oil production (million bbl/yr)',
    'gas_million_bbleq': 'Gas production (million bbl-eq/yr)',
    'energy_million_bbleq': 'Total energy production (million bbl-eq/yr)',
    'oil_gas_ratio': 'Oil to gas ratio (J/J)',
    'winterozone_exceed_num': 'Winter ozone exceedances (number of days/yr)',
    'producing_wells': 'Total producing wells (count)',
    'newwells': 'New wells (count)',
    'prodfromhighwells_thousbbleq_mnth': 'High-yield well production (thousand bbl-eq/month)',
    'prodfromlowwells_thousbbleq_mnth': 'Low-yield well production (thousand bbl-eq/month)',
    'prodfromoilwells_thousbbleq_mnth': 'Oil well production (thousand bbl-eq/month)',
    'prodfromgaswells_thousbbleq_mnth': 'Gas well production (thousand bbl-eq/month)',
    'pcnt_prodfromhighwells': 'Percentage of total production from high-yield wells (%)',
    'gaswells': 'Gas wells (count)',
    'oilwells': 'Oil wells (count)',
    'new_gaswells': 'New gas wells (number of new wells in the year)',
    'new_oilwells': 'New oil wells (number of new wells in the year)',
    'highprodwells': 'High-yield wells (count)',
    'lowprodwells': 'Low-yield wells (count)',
    'emissinens_totenergy': 'Total energy emissions intensity (J emitted/J total energy produced)',
    'emissintens_gas': 'gas emissions intensity (J emitted/J gas produced)',
    'emisCO2eq20_millnMg': 'Basin-wide methane emissions as CO2 equiv. (million Mg/yr; 20-yr global warming potential)', 
    'Utahgasprice_dollperMCF': 'Average industrial natural gas price in Utah ($/MCF)',
    'Utahcrudeprice_dollperBBL': 'Average first purchase crude oil price in Utah ($/bbl)'
}

def get_display_name(col_name):
    """Get display name for a column, fallback to formatted original name"""
    return display_names.get(col_name, col_name.replace('_', ' ').title())



def create_time_series_plot(df, selected_metrics):
    """Create interactive time series plot"""
    if not selected_metrics:
        return None
        
    fig = make_subplots(
        rows=len(selected_metrics), cols=1,
        #subplot_titles=[metric.replace('_', ' ').title() for metric in selected_metrics],
        subplot_titles=[get_display_name(metric) for metric in selected_metrics],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, metric in enumerate(selected_metrics):
        if metric in df.columns:
            # Remove null values for plotting
            mask = df[metric].notna()
            x_data = df.loc[mask, 'year']
            y_data = df.loc[mask, metric]
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{get_display_name(metric)}</b><br>' +
                                'Year: %{x}<br>' +
                                'Value: %{y:.2f}<br>' +
                                '<extra></extra>'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=200 * len(selected_metrics) + 100,
        title_text="",
        title_x=0.5,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Year", row=len(selected_metrics), col=1)
    
    return fig

def create_correlation_heatmap(df, selected_metrics=None):
    """Create correlation heatmap"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'year']
    
    # If selected metrics are provided, filter to only those
    if selected_metrics:
        numeric_cols = [col for col in selected_metrics if col in numeric_cols]
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Create display names for the correlation matrix
        display_labels = [get_display_name(col) for col in corr_matrix.columns]
        
        fig = px.imshow(
            corr_matrix,
            x=display_labels,
            y=display_labels,
            color_continuous_scale='RdBu',  # Removed the '_r' to fix the color scheme
            aspect='auto',
            title="🔗 Correlation matrix for variables selected in the time series",
            zmin=-1,  # Ensure the scale goes from -1 to 1
            zmax=1
        )
        
        # Add correlation values as text with improved color logic
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[j, i]
                
                # Improved text color logic
                # For RdBu colorscale: strong positive (red) and strong negative (blue) need white text
                # Weak correlations (white/light colors) need black text
                if abs(corr_value) > 0.6:  # Lowered threshold since colors get dark sooner
                    text_color = "white"
                else:
                    text_color = "black"
                
                fig.add_annotation(
                    x=i, y=j,
                    text=str(round(corr_value, 2)),
                    showarrow=False,
                    font=dict(color=text_color, size=12)  # Also made font slightly larger
                )
        
        fig.update_layout(height=600)
        return fig
    return None

def create_scatter_plot(df, x_metric, y_metric):
    """Create interactive scatter plot between two selected metrics"""
    if x_metric not in df.columns or y_metric not in df.columns:
        return None
    
    # Remove rows where either metric is null
    mask = df[x_metric].notna() & df[y_metric].notna()
    plot_data = df[mask]
    
    if len(plot_data) == 0:
        return None
    
    fig = px.scatter(
        plot_data,
        x=x_metric,
        y=y_metric,
        text='year',  # Add this to show year labels on points
        hover_data=['year'],
        title=f"📊 {get_display_name(x_metric)} vs {get_display_name(y_metric)}",
        labels={
            x_metric: get_display_name(x_metric),
            y_metric: get_display_name(y_metric)
        }
    )
    
    # Add simple linear trend line using numpy
    x_data = plot_data[x_metric].values
    y_data = plot_data[y_metric].values
    
    # Calculate linear regression using numpy
    coefficients = np.polyfit(x_data, y_data, 1)  # 1 = linear
    trend_line = np.poly1d(coefficients)
    
    # Create trend line data points
    x_trend = np.linspace(x_data.min(), x_data.max(), 100)
    y_trend = trend_line(x_trend)
    
    # Add trend line to the plot
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2),
            hovertemplate='Trend Line<extra></extra>'
        )
    )
    
    # Customize the scatter plot
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, color='#1f77b4'),
        selector=dict(mode='markers')
    )
    
    # Customize text labels - small font and position to avoid overlap
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10, color="black"),
        selector=dict(mode='markers+text')
    )
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        showlegend=False  # Hide legend since we only have one trend line
    )
    
    return fig

def create_distribution_plot(df, selected_metric):
    """Create distribution plot for selected metric"""
    if selected_metric in df.columns:
        data = df[selected_metric].dropna()
        
        if len(data) == 0:
            return None
            
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Distribution', 'Box Plot'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=min(10, len(data)//2),
                name="Distribution",
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data,
                name="Box Plot",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"📊 Distribution Analysis: {get_display_name(selected_metric)}",
            showlegend=False,
            height=400
        )
        
        return fig
    return None

def main():
    """Main application"""
    
    # Add Google Analytics tracking first (before any content)
    add_google_analytics()
    
    # Integrated header with logo and title
    st.image("https://www.usu.edu/binghamresearch/images/logos/BRS_01_UStateLeft_AggieBlue.png", width=600)
    st.markdown('<h1 style="font-size: 3rem; font-weight: bold; color: #1e3a5f; margin-bottom: 2rem; text-align: left;">Uinta Basin Energy and Emissions Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("---")  # Add a divider line
    
    # Explanation section
    st.markdown("""
    This Explorer contains annual emissions, air quality, and oil and gas activity data for the Uinta Basin, Utah. 
    Emissions and air quality data were compiled by the [USU Bingham Research Center](https://www.usu.edu/binghamresearch/dataexplorer).
    Oil and gas activity data are from the [Utah Division of Oil, Gas and Mining](https://ogm.utah.gov/). Oil and gas price data are from [eia.gov](https://www.eia.gov/).
    Trends in air emissions data may be due to changes in technologies, regulations, or standard indusry practices.
    
    We note the following major regulatory rulemakings and other significant actions that may have impacted emissions trends:
    * 2012-2015: CFR40, Part 60, Subpart OOOO, regulating organic compound emissions from new or modified oil and gas facilities, was phased in by EPA
    * 2016: CFR40, Part 60, Subpart OOOOa, requiring periodic leak detection and repair at many facilities, was implemented.
    * 2018: A [survey of operators](https://online.ucpress.edu/elementa/article/doi/10.1525/elementa.381/112514/Aerial-and-ground-based-optical-gas-imaging-survey) found that 83% performed leak detection and repair at all their facilities at least annually.
    * 2024: Full compliance with the EPA Federal Implementation Plan for Indian Country Lands of the Uintah and Ouray Indian Reservation was required.
    
    Detailed information about the data used in this app is available at https://www.usu.edu/binghamresearch/dataexplorer.
    """)

    st.markdown("---")  # Add another divider line
    
    # Load data (no file upload needed)
    df = load_data()
    st.success("✅ Data loaded successfully!")
    
    # Show data source info
    #st.info("📊 Analyzing energy production and environmental data for the Uinta Basin, Utah")
    
    # Metric selection setup
    numeric_columns = [col for col in df.columns if col != 'year' and df[col].dtype in ['float64', 'int64']]
    
    # Default selection - specify the three variables you want
    desired_defaults = ['energy_million_bbleq', 'basinwide_ch4_emiss_Mg_hr', 'emissinens_totenergy']
    default_metrics = [col for col in desired_defaults if col in numeric_columns]
    
    # Controls for time series
    st.subheader("📈 Time Series and Correlation Analysis Controls")
    
    # Create two columns for the controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Year range selector
        year_range = st.slider(
            "Select Year Range",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=(int(df['year'].min()), int(df['year'].max()))
        )
    
    with col2:
        # Variable selection for time series
        selected_metrics = st.multiselect(
            "Select Metrics for Time Series",
            options=numeric_columns,
            default=default_metrics,
            format_func=get_display_name
        )
    
    # Filter data by year range
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Main content
    if not filtered_df.empty:
        # Time series plot
        if selected_metrics:
            st.subheader("📈 Time Series")
            
            # Add expandable instructions
            with st.expander("ℹ️ Time Series Instructions"):
                st.markdown("""
                **How to use the Time Series plot:**
                
                - **Select Variables**: Use the "Select Metrics for Time Series" dropdown above to choose which variables to display
                - **Adjust Time Range**: Use the year slider to focus on specific time periods
                - **Interactive Features**:
                    - Hover over data points to see exact values
                    - Each variable is plotted in its own subplot with its own scale
                    - The plots share the same x-axis (years) for easy comparison
                
                **Interpreting the Data:**
                - Look for trends over time in each variable
                - Compare timing of changes across different variables
                - Note any correlations between variables (peaks and valleys at similar times)
                - Consider external factors that might explain trends (regulations, market conditions, etc.)
                
                **Tips:**
                - Start with 2-3 variables to avoid overcrowding
                - Use the correlation matrix or scatter plot below to identify strongly related variables
                - Adjust the year range to focus on periods of interest
                """)
            
            fig_ts = create_time_series_plot(filtered_df, selected_metrics)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
        
        # Correlation heatmap (show on demand)
        st.subheader("🔗 Correlation Analysis")
        show_correlation = st.checkbox("Show correlation matrix for selected variables")

        if show_correlation:
            # Add expandable instructions (only when correlation is enabled)
            with st.expander("ℹ️ Correlation Matrix Instructions"):
                st.markdown("""
                **How to interpret the Correlation Matrix:**
                
                - **Color Scale**: Colors range from dark blue (strong negative correlation) to dark red (strong positive correlation)
                - **Numbers in cells**: Show the exact correlation coefficient (-1.0 to +1.0)
                - **Diagonal**: Always shows 1.0 (perfect correlation of a variable with itself)
                
                **Understanding Correlation Values:**
                - **+0.7 to +1.0**: Strong positive correlation (as one increases, the other tends to increase)
                - **+0.3 to +0.7**: Moderate positive correlation
                - **-0.3 to +0.3**: Weak or no linear correlation
                - **-0.7 to -0.3**: Moderate negative correlation
                - **-1.0 to -0.7**: Strong negative correlation (as one increases, the other tends to decrease)
                - These are Pearson r values
                
                **Tips for Analysis:**
                - Look for dark red or dark blue cells to identify strong relationships
                - Consider potential causal relationships (correlation ≠ causation)
                - Strong correlations may indicate variables that move together due to common drivers
                - Use this to guide your scatter plot analysis below
                
                **Note**: Only variables selected in the time series above are included in this matrix.
                """)
            
            fig_corr = create_correlation_heatmap(filtered_df, selected_metrics)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            
        # Scatter plot analysis
        st.subheader("📊 Scatter Plot Analysis")

        # Add expandable instructions
        with st.expander("ℹ️ Scatter Plot Instructions"):
            st.markdown("""
            **How to use the Scatter Plot:**
            
            - **Variable Selection**: Choose different variables for X and Y axes using the dropdowns below
            - **Interactive Features**:
                - Hover over points to see exact values and year labels
                - Each point represents one year of data
                - Year labels are displayed on each data point
            
            **Understanding the Plot:**
            - **Data Points**: Each dot represents one year, showing the relationship between two variables
            - **Red Trend Line**: Shows the overall linear relationship (ordinary least squares regression)
            - **Correlation Coefficient**: Displayed below the plot (Pearson r value; ranges from -1 to +1)
            
            **Interpreting Relationships:**
            - **Upward trend**: Positive relationship (as X increases, Y tends to increase)
            - **Downward trend**: Negative relationship (as X increases, Y tends to decrease)
            - **Scattered points with no obvious trend**: Weak or no linear relationship
            - **Points close to the red trend line**: Strong linear relationship
            
            **Analysis Tips:**
            - Compare the correlation coefficient with what you see visually
            - Look for outlier years that don't follow the general pattern
            - Consider time-based trends (are recent years clustered differently?)
            - Use insights from the correlation matrix to guide variable selection
            
            **Note**: Plots only include data for the year range selected in the time series above.
            """)

        # Create two columns for x and y variable selection
        col1, col2 = st.columns(2)

        with col1:
            x_variable = st.selectbox(
                "Select X-axis variable:",
                options=numeric_columns,
                index=numeric_columns.index('gas_million_bbleq') if 'gas_million_bbleq' in numeric_columns else 0,
                format_func=get_display_name
            )

        with col2:
            y_variable = st.selectbox(
                "Select Y-axis variable:",
                options=numeric_columns,
                index=numeric_columns.index('basinwide_ch4_emiss_Mg_hr') if 'basinwide_ch4_emiss_Mg_hr' in numeric_columns else 1,
                format_func=get_display_name
            )

        # Create and display scatter plot
        if x_variable != y_variable:
            fig_scatter = create_scatter_plot(filtered_df, x_variable, y_variable)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Calculate and display correlation coefficient and regression info
                correlation = filtered_df[x_variable].corr(filtered_df[y_variable])
                if not pd.isna(correlation):
                    st.info(f"📈 Correlation coefficient (Pearson r): {correlation:.3f} | 🔴 Red line: OLS linear regression")
        else:
            st.warning("⚠️ Please select different variables for X and Y axes.")
        
        # Distribution Analysis
        st.subheader("📊 Distribution Analysis")

        # Checkbox to enable distribution analysis
        show_distribution = st.checkbox("Show distribution analysis for a selected variable")

        if show_distribution:
            # Add expandable instructions (only when distribution analysis is enabled)
            with st.expander("ℹ️ Distribution Analysis Instructions"):
                st.markdown("""
                **How to use the Distribution Analysis:**
                
                - **Variable Selection**: Choose any variable from the dropdown to analyze its distribution
                - **Two Views**: The analysis shows both a histogram and box plot side by side
                
                **Understanding the Histogram (Left Plot):**
                - **Bars**: Show frequency of values in different ranges (bins)
                - **Shape**: Reveals the distribution pattern:
                    - **Normal**: Bell-shaped, symmetric with the maximum at the center
                    - **Skewed**: Long tail on one side (left or right skewed)
                    - **Uniform**: Relatively flat across all values
                    - **Bi- or multi-modal**: Two or more distinct peaks
                
                **Understanding the Box Plot (Right Plot):**
                - **Box**: Shows the middle 50% of data (interquartile range)
                - **Line in box**: Median (middle value)
                - **Whiskers**: Extend to show data range (1.5 × IQR)
                - **Dots**: Outliers (values far from the typical range)
                
                **Key Statistics Explained:**
                - **Mean**: Average value (sensitive to outliers)
                - **Median**: Middle value when sorted (robust to outliers)
                - **Std Dev**: Measure of spread (higher relative to mean = more variability)
                - **Range**: Difference between maximum and minimum values
                
                **Analysis Tips:**
                - Compare mean vs median: if very different, data may be skewed
                - Look for outliers that might need special attention
                - Consider if the distribution shape matches your expectations
                - Use this to understand data quality and variability
                
                **Note**: Plots only include data for the year range selected in the time series above.
                """)
            
            # Variable selection for distribution
            distribution_variable = st.selectbox(
                "Select variable for distribution analysis:",
                options=numeric_columns,
                index=numeric_columns.index('energy_million_bbleq') if 'energy_million_bbleq' in numeric_columns else 0,
                format_func=get_display_name,
                key="distribution_selector"  # Unique key to avoid conflicts
            )
            
            # Create and display distribution plot
            fig_dist = create_distribution_plot(filtered_df, distribution_variable)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Add some basic statistics
                data = filtered_df[distribution_variable].dropna()
                if len(data) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{data.mean():.2f}")
                    with col2:
                        st.metric("Median", f"{data.median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{data.std():.2f}")
                    with col4:
                        st.metric("Range", f"{data.max() - data.min():.2f}")
        
        # Data table
        st.subheader("📋 Entire Dataset")
        show_raw_data = st.checkbox("Show raw data")
        if show_raw_data:
            st.dataframe(filtered_df, use_container_width=True)
        
        # Download processed data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"energy_data_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )
        
        # Statistical summary
        with st.expander("📈 Statistical Summary"):
            # Add instructions dropdown within the statistical summary expander
            with st.expander("ℹ️ Statistical Summary Instructions"):
                st.markdown("""
                **Understanding the Statistical Summary Table:**
                
                **Rows Explained:**
                - **count**: Number of non-missing values for each variable
                - **mean**: Average value across all years
                - **std**: Standard deviation (measure of variability around the mean)
                - **min**: Minimum (lowest) value observed
                - **25%**: First quartile (25% of values are below this)
                - **50%**: Median (middle value, 50% of values are above and below)
                - **75%**: Third quartile (75% of values are below this)
                - **max**: Maximum (highest) value observed
                
                **How to Interpret:**
                - **Large std relative to mean**: High variability in the data
                - **Mean vs Median**: If very different, data may be skewed
                - **Quartiles (25%, 50%, 75%)**: Show how data is distributed
                - **Range (max - min)**: Shows the full span of values
                
                **Analysis Tips:**
                - Compare means across variables to understand relative scales
                - Look for variables with high standard deviations relative to their means (more volatile)
                - Use quartiles to identify typical ranges for each variable
                - Check if minimums/maximums seem reasonable or might be outliers
                - Consider which variables show the most/least variation over time
                
                **Note**: Statistics are calculated only for the year range selected in the time series above.
                """)
            
            # Create a copy of the dataframe with display names
            summary_df = filtered_df.describe()
            
            # Rename columns to use display names
            renamed_columns = {}
            for col in summary_df.columns:
                if col != 'year':  # Skip year column or include it if you want
                    renamed_columns[col] = get_display_name(col)
            
            summary_df = summary_df.rename(columns=renamed_columns)
            st.write(summary_df)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Explorer** | Built with Streamlit and Plotly")


if __name__ == "__main__":
    main()
    
