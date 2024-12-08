import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load Dataset
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Filter Data Based on User Inputs
def filter_data(df, goals=None, assists=None, age_range=None, position=None, metrics=None):
    filtered_df = df.copy()
    if goals is not None:
        filtered_df = filtered_df[filtered_df['GoalsScored'] >= goals]
    if assists is not None:
        filtered_df = filtered_df[filtered_df['Assists'] >= assists]
    if age_range:
        min_age, max_age = age_range
        filtered_df = filtered_df[(filtered_df['Age'] >= min_age) & (filtered_df['Age'] <= max_age)]
    if position:
        filtered_df = filtered_df[filtered_df['Position'].isin(position)]
    if metrics:
        for metric, threshold in metrics.items():
            filtered_df = filtered_df[filtered_df[metric] >= threshold]
    return filtered_df

# Generate PDF Report
def generate_pdf_report(df):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Scouting Report", ln=True, align='C')
    for index, row in df.iterrows():
        pdf.ln(10)
        for col, value in row.items():
            pdf.cell(200, 10, txt=f"{col}: {value}", ln=True)
    return pdf.output(dest="S").encode("latin1")

# Visualization Functions
def plot_radar_chart(df, player_name):
    categories = ['PhysicalStrength', 'Speed', 'TechnicalAbility', 'PassAccuracy']
    values = df[df['Name'] == player_name][categories].values.flatten().tolist()
    values += values[:1]  # Close the circle
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    st.pyplot(fig)

def plot_comparison_bar_chart(df, players, metrics):
    filtered_df = df[df['Name'].isin(players)]
    melted_df = filtered_df.melt(id_vars='Name', value_vars=metrics)
    sns.barplot(data=melted_df, x='variable', y='value', hue='Name')
    plt.title("Player Comparison")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Main Application
def main():
    st.title("Scouting Report Generator")
    st.sidebar.header("Filters")

    # Load Dataset
    file_path = r'C:\Users\kbust\Desktop\aifinalproj\soccer_scouting_dataset.csv'
    data = load_data(file_path)

    # Filter Inputs
    goals = st.sidebar.slider("Minimum Goals Scored", 0, int(data['GoalsScored'].max()), 0)
    assists = st.sidebar.slider("Minimum Assists", 0, int(data['Assists'].max()), 0)
    age_range = st.sidebar.slider("Age Range", int(data['Age'].min()), int(data['Age'].max()), (18, 30))
    position = st.sidebar.multiselect("Preferred Position", options=data['Position'].unique())
    metrics = {
        "PassAccuracy": st.sidebar.slider("Pass Accuracy", 0, 100, 50),
        "PhysicalStrength": st.sidebar.slider("Physical Strength", 0, 100, 50),
        "Speed": st.sidebar.slider("Speed", 0, 100, 50),
        "TechnicalAbility": st.sidebar.slider("Technical Ability", 0, 100, 50),
    }

    # Filter Data
    filtered_data = filter_data(data, goals, assists, age_range, position, metrics)
    st.subheader("Filtered Players")
    st.write(filtered_data)

    # Player Comparison
    st.subheader("Player Comparison")
    selected_players = st.multiselect("Select Players for Comparison", options=filtered_data['Name'].unique())
    if selected_players:
        plot_comparison_bar_chart(filtered_data, selected_players, ['GoalsScored', 'Assists', 'MinutesPlayed'])

    # Radar Chart
    st.subheader("Radar Chart for Player")
    selected_player = st.selectbox("Select a Player", options=filtered_data['Name'].unique())
    if selected_player:
        plot_radar_chart(filtered_data, selected_player)

    # Generate PDF Report
    st.subheader("Export Report")
    if st.button("Download CSV"):
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="scouting_report.csv", mime='text/csv')

    if st.button("Download PDF"):
        pdf = generate_pdf_report(filtered_data)
        st.download_button("Download PDF", data=pdf, file_name="scouting_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
