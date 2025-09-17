import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#  Page Config 
st.set_page_config(page_title="Tourism Recommender", layout="wide")

#  Custom CSS Styling 
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            color: #3B82F6;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .description {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 2em;
        }
        .highlight-box {
            background-color: #F1F5F9;
            padding: 1em;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

#  Title 
st.markdown('<div class="main-title">🌍 Tourism Data Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Analyze trends, predict visit modes, and get personalized attraction suggestions</div>', unsafe_allow_html=True)

# Sidebar File Upload 
with st.sidebar:
    st.header("📂 Load Dataset")
    use_local = st.toggle("Use default file", value=True)
    df = None

    if use_local:
        try:
            df = pd.read_csv(r"C:\Users\Ranjitha\OneDrive\Documents\Tourism_Project.csv")
            st.success("Loaded default dataset.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")

#  Tabs 
if df is not None:
    df.columns = df.columns.str.strip()
    required = ['VisitMode', 'CityName', 'Region', 'Country', 'Continent', 'Attraction', 'Rating']
    if not all(col in df.columns for col in required):
        st.error("⚠️ Required columns are missing.")
    else:
        df = df.dropna(subset=required)

        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Prediction", "🎯 Recommendations"])

        # TAB 1: Dashboard 
        with tab1:
            st.subheader("📈 Tourism Overview")

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='highlight-box'><h2>{df['UserId'].nunique()}</h2><p>Unique Users</p></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='highlight-box'><h2>{df['Attraction'].nunique()}</h2><p>Attractions</p></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='highlight-box'><h2>{df['VisitMode'].nunique()}</h2><p>Visit Modes</p></div>", unsafe_allow_html=True)

            with st.expander("📊 Visit Mode Distribution"):
                fig, ax = plt.subplots()
                df['VisitMode'].value_counts().plot(kind='bar', color='#60A5FA', ax=ax)
                ax.set_ylabel("Count")
                ax.set_title("Visit Mode Frequency")
                st.pyplot(fig)

            with st.expander("⭐ Top Rated Attractions"):
                top_rated = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)
                fig2, ax2 = plt.subplots()
                top_rated.plot(kind='barh', color='#F59E0B', ax=ax2)
                ax2.set_xlabel("Avg Rating")
                st.pyplot(fig2)

            col4, col5 = st.columns(2)

            with col4:
                with st.expander("📈 Visit Trends Over Time (Line Chart)"):
                    if 'VisitYear' in df.columns and 'VisitMonth' in df.columns:
                        df['VisitMonth'] = df['VisitMonth'].astype(str).str.zfill(2)
                        df['VisitYearMonth'] = df['VisitYear'].astype(str) + '-' + df['VisitMonth']
                        trend_df = df.groupby('VisitYearMonth').size().sort_index()
                        fig5, ax5 = plt.subplots(figsize=(4, 3))
                        trend_df.plot(kind='line', marker='o', color='#10B981', ax=ax5)
                        ax5.set_xlabel("Year-Month")
                        ax5.set_ylabel("Number of Visits")
                        ax5.set_title("Monthly Visit Trends")
                        ax5.tick_params(axis='x', rotation=45)
                        st.pyplot(fig5)
                    else:
                        st.warning("VisitYear and VisitMonth columns are required for line chart.")

            with col5:
                with st.expander("📊 Rating Distribution (Histogram)"):
                    fig6, ax6 = plt.subplots(figsize=(4, 3))
                    df['Rating'].dropna().astype(float).plot(kind='hist', bins=10, color='#F87171', edgecolor='black', ax=ax6)
                    ax6.set_title("Rating Distribution")
                    ax6.set_xlabel("Rating")
                    st.pyplot(fig6)

        #  TAB 2: Prediction 
        with tab2:
            st.subheader("🔮 Predict Visit Mode")
            df_input = df[['Continent', 'Country', 'Region', 'CityName', 'VisitMode']].dropna().astype(str)

            continent = st.selectbox("Continent", sorted(df_input['Continent'].unique()))
            country = st.selectbox("Country", sorted(df_input[df_input['Continent'] == continent]['Country'].unique()))
            region = st.selectbox("Region", sorted(df_input[df_input['Country'] == country]['Region'].unique()))
            city = st.selectbox("City", sorted(df_input[df_input['Region'] == region]['CityName'].unique()))

            le1, le2, le3, le4, le_y = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
            df_input['Continent'] = le1.fit_transform(df_input['Continent'])
            df_input['Country'] = le2.fit_transform(df_input['Country'])
            df_input['Region'] = le3.fit_transform(df_input['Region'])
            df_input['CityName'] = le4.fit_transform(df_input['CityName'])
            df_input['VisitMode'] = le_y.fit_transform(df_input['VisitMode'])

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(df_input[['Continent', 'Country', 'Region', 'CityName']], df_input['VisitMode'])

            user_input = pd.DataFrame([[
                le1.transform([continent])[0],
                le2.transform([country])[0],
                le3.transform([region])[0],
                le4.transform([city])[0]
            ]], columns=['Continent', 'Country', 'Region', 'CityName'])

            predicted_mode = le_y.inverse_transform(model.predict(user_input))[0]
            st.success(f"🎉 Predicted Visit Mode: **{predicted_mode}**")

        #  TAB 3: Recommendations 
        with tab3:
            st.subheader("🎯 Personalized Recommendations")

            filtered_df = df[(df['CityName'].astype(str) == city) & (df['VisitMode'].astype(str) == predicted_mode)]
            if not filtered_df.empty:
                st.markdown("**Top 5 Attractions Matching Your Profile:**")
                recommendations = filtered_df[['Attraction', 'Rating']].drop_duplicates().sort_values(by='Rating', ascending=False)
                st.dataframe(recommendations.head(5), use_container_width=True)

                st.markdown("**Full Recommendations Data:**")
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.warning("No recommendations available for this profile.")
