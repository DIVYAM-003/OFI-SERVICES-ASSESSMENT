import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import glob
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="NexGen Predictive Optimizer",
    page_icon="🚀",
    layout="wide"
)

# --- Caching ---
# Cache data loading
@st.cache_data
def load_all_data():
    """Loads all 7 CSVs from the 'data' directory and merges them."""
    data = {}
    
    # Check if 'data' directory exists
    if not os.path.isdir('data'):
        st.error("Error: 'data' directory not found.")
        st.info("Please create a 'data' folder and place the 7 CSV files inside it.")
        return None, None

    data_files = glob.glob("data/*.csv")
    
    if len(data_files) < 7:
        st.warning(f"Warning: Found {len(data_files)}/7 CSV files in the 'data' directory. Some features may be missing.")
        if not data_files:
            return None, None
            
    for f in data_files:
        try:
            key = os.path.basename(f).replace('.csv', '')
            data[key] = pd.read_csv(f)
        except Exception as e:
            st.error(f"Error loading {f}: {e}")
            return None, None

    # Create the master merged dataframe for analysis
    try:
        # Core data for modeling and dashboard
        master_df = data['orders'].merge(data['delivery_performance'], on='Order_ID')
        master_df = master_df.merge(data['routes_distance'], on='Order_ID')
        
        # Add customer feedback
        if 'customer_feedback' in data:
            master_df = master_df.merge(data['customer_feedback'], on='Order_ID', how='left')
            
        return master_df, data
    except KeyError:
        st.error("Error: Could not merge key dataframes. Make sure 'orders.csv', 'delivery_performance.csv', and 'routes_distance.csv' are present.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during data merging: {e}")
        return None, None

# Cache model loading
@st.cache_resource
def load_model(model_path='delivery_model.pkl'):
    """Loads the trained ML model from disk."""
    if not os.path.exists(model_path):
        st.error("Error: 'delivery_model.pkl' not found.")
        st.info("Please run `python train_model.py` in your terminal to create the model file.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Load Data and Model ---
master_df, all_data = load_all_data()
model = load_model()

if master_df is None:
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("NexGen Logistics 🚀")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Your Tool",
    ("Performance Dashboard", "Predictive Optimizer")
)
st.sidebar.markdown("---")
st.sidebar.info("This application is a prototype for the NexGen Logistics Innovation Challenge.")

# --- Page 1: Performance Dashboard ---
if page == "Performance Dashboard":
    st.title("🚚 Delivery Performance Dashboard")
    st.markdown("A high-level overview of our current logistics performance.")

    # --- KPIs ---
    st.header("Key Performance Indicators (KPIs)")
    
    # Calculate KPIs
    total_orders = master_df.shape[0]
    on_time_df = master_df[master_df['Delivery_Status'] == 'On-Time']
    on_time_pct = (on_time_df.shape[0] / total_orders) * 100
    avg_rating = master_df['Customer_Rating'].mean()
    avg_delivery_cost = master_df['Delivery_Cost_INR'].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Orders", f"{total_orders:,}")
    kpi2.metric("On-Time Delivery Rate", f"{on_time_pct:.2f}%")
    kpi3.metric("Avg. Customer Rating", f"{avg_rating:.2f} ★")
    kpi4.metric("Avg. Delivery Cost", f"₹{avg_delivery_cost:.2f}")
    
    st.markdown("---")

    # --- Visualizations ---
    st.header("Performance Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Delivery Status Breakdown
        st.subheader("Delivery Status Breakdown")
        status_counts = master_df['Delivery_Status'].value_counts().reset_index()
        status_counts.columns = ['Delivery_Status', 'Count']
        fig_pie = px.pie(status_counts, 
                         names='Delivery_Status', 
                         values='Count', 
                         title="Overall Delivery Status",
                         color_discrete_map={'On-Time': 'green', 'Slightly-Delayed': 'orange', 'Severely-Delayed': 'red'})
        st.plotly_chart(fig_pie, use_container_width=True)

        # Chart 3: Delays by Product Category
        st.subheader("Delays by Product Category")
        fig_bar_prod = px.histogram(master_df, 
                                    x='Product_Category', 
                                    color='Delivery_Status', 
                                    barmode='group',
                                    title="Delivery Status by Product Category")
        st.plotly_chart(fig_bar_prod, use_container_width=True)

    with col2:
        # Chart 2: Performance by Carrier
        st.subheader("Performance by Carrier")
        fig_bar_carrier = px.histogram(master_df, 
                                       x='Carrier', 
                                       color='Delivery_Status', 
                                       barmode='group',
                                       title="Delivery Status by Carrier",
                                       category_orders={"Delivery_Status": ["On-Time", "Slightly-Delayed", "Severely-Delayed"]})
        st.plotly_chart(fig_bar_carrier, use_container_width=True)

        # Chart 4: Customer Rating vs. Delivery Status
        st.subheader("Customer Rating vs. Delivery Status")
        fig_box = px.box(master_df, 
                         x='Delivery_Status', 
                         y='Customer_Rating', 
                         color='Delivery_Status',
                         title="Customer Rating by Delivery Status")
        st.plotly_chart(fig_box, use_container_width=True)


# --- Page 2: Predictive Optimizer ---
elif page == "Predictive Optimizer":
    st.title("🔮 Predictive Delivery Optimizer")
    st.markdown("Input new order details to predict the probability of a delay.")

    if model is None:
        st.error("Model not loaded. Cannot proceed with prediction.")
        st.stop()
        
    # --- Input Form ---
    st.header("Enter Order Details")
    
    # Get unique values for dropdowns from the master dataframe
    # This ensures the inputs are valid for the model
    customer_segments = master_df['Customer_Segment'].unique()
    priorities = master_df['Priority'].unique()
    product_categories = master_df['Product_Category'].unique()
    origins = master_df['Origin'].unique()
    destinations = master_df['Destination'].unique()
    special_handlings = master_df['Special_Handling'].unique()
    carriers = master_df['Carrier'].unique()
    weather_impacts = master_df['Weather_Impact'].unique()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Column 1
        with col1:
            customer_segment = st.selectbox("Customer Segment", customer_segments, help="Select the customer segment.")
            priority = st.selectbox("Priority", priorities, help="Select the order priority.")
            product_category = st.selectbox("Product Category", product_categories, help="Select the product category.")
            promised_days = st.number_input("Promised Delivery (Days)", min_value=1, max_value=20, value=5, help="Number of days promised to the customer.")

        # Column 2
        with col2:
            origin = st.selectbox("Origin Warehouse", origins, help="Select the origin warehouse or city.")
            destination = st.selectbox("Destination City", destinations, help="Select the destination city.")
            carrier = st.selectbox("Carrier", carriers, help="Select the carrier for this delivery.")
            distance_km = st.number_input("Distance (KM)", min_value=10, max_value=6000, value=500, help="Total distance of the route.")

        # Column 3
        with col3:
            special_handling = st.selectbox("Special Handling", special_handlings, help="Any special handling requirements.")
            traffic_delay = st.number_input("Est. Traffic Delay (Mins)", min_value=0, max_value=300, value=30, help="Estimated delay from traffic (in minutes).")
            weather_impact = st.selectbox("Weather Impact", weather_impacts, help="Select the current weather impact.")
        
        st.markdown("---")
        submit_button = st.form_submit_button(label="Analyze Delay Risk", type="primary")

    # --- Prediction Logic ---
    if submit_button:
        # 1. Create the input dataframe for the model
        # The column names MUST match the 'FEATURES' list from train_model.py
        input_data = pd.DataFrame({
            'Customer_Segment': [customer_segment],
            'Priority': [priority],
            'Product_Category': [product_category],
            'Origin': [origin],
            'Destination': [destination],
            'Special_Handling': [special_handling],
            'Promised_Delivery_Days': [promised_days],
            'Carrier': [carrier],
            'Distance_KM': [distance_km],
            'Traffic_Delay_Minutes': [traffic_delay],
            'Weather_Impact': [weather_impact]
        })

        st.subheader("Analyzing Order...")
        
        try:
            # 2. Predict the probability
            # model.predict_proba() returns a list of [prob_on_time, prob_delayed]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Get the probability of 'Delayed' (which is the class '1')
            delay_probability = prediction_proba[1]
            on_time_probability = prediction_proba[0]

            # 3. Display the result
            st.header("Prediction Result")
            
            result_col1, result_col2 = st.columns([1,2])

            with result_col1:
                if delay_probability > 0.6:
                    st.error(f"High Risk: {delay_probability*100:.2f}%")
                elif delay_probability > 0.3:
                    st.warning(f"Medium Risk: {delay_probability*100:.2f}%")
                else:
                    st.success(f"Low Risk: {delay_probability*100:.2f}%")
                
                # Plotly Gauge Chart
                import plotly.graph_objects as go
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = delay_probability,
                    number = {'valueformat': '.1%'},
                    title = {'text': "Probability of Delay"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 0.3], 'color': 'green'},
                            {'range': [0.3, 0.6], 'color': 'orange'},
                            {'range': [0.6, 1], 'color': 'red'}],
                    }
                ))
                fig_gauge.update_layout(height=250, margin={'t':0, 'b':0, 'l':0, 'r':0})
                st.plotly_chart(fig_gauge, use_container_width=True)


            with result_col2:
                st.markdown(f"**Probability of being On-Time:** `{on_time_probability*100:.2f}%`")
                st.markdown(f"**Probability of being Delayed:** `{delay_probability*100:.2f}%`")

                if delay_probability > 0.6:
                    st.error("**Recommended Action: HIGH RISK**")
                    st.markdown("""
                        * **Immediate Review:** Flag this order for manual review.
                        * **Carrier Check:** Consider switching to a more reliable carrier (e.g., check dashboard for best performers on this route).
                        * **Proactive Communication:** Alert the customer *now* of a potential delay to manage expectations.
                        * **Expedite:** Prioritize this order in the warehouse for faster handling.
                    """)
                elif delay_probability > 0.3:
                    st.warning("**Recommended Action: MEDIUM RISK**")
                    st.markdown("""
                        * **Monitor:** Keep this order on a "watch list."
                        * **Verify:** Double-check that all order details (e.g., address, special handling) are correct.
                        * **Prepare:** Have a customer service template ready in case the delay occurs.
                    """)
                else:
                    st.success("**Recommended Action: LOW RISK**")
                    st.markdown("""
                        * **Standard Procedure:** No immediate action required.
                        * **Confirmation:** Proceed with standard logistics operations.
                    """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure the model file is correct and the input data is valid.")
