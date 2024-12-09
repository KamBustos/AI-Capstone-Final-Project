import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
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

# Train ML Model and Predict Player Potential
def train_ml_model(df):
    features = ['GoalsScored', 'Assists', 'MinutesPlayed', 'PassAccuracy', 
                'PhysicalStrength', 'Speed', 'TechnicalAbility']
    target = 'PotentialRating'
    X = df[features]
    y = (df[target] > 80).astype(int)  # Binary classification: High Potential (1), Low Potential (0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report

# Predict Player Potential
def predict_potential(model, player_data):
    features = ['GoalsScored', 'Assists', 'MinutesPlayed', 'PassAccuracy', 
                'PhysicalStrength', 'Speed', 'TechnicalAbility']
    prediction = model.predict(player_data[features])
    return prediction

# Visualize Classification Metrics
def plot_metrics(report):
    # Extract metrics
    classes = ["Low Potential (0)", "High Potential (1)"]
    precision = [report["0"]["precision"], report["1"]["precision"]]
    recall = [report["0"]["recall"], report["1"]["recall"]]
    f1_score = [report["0"]["f1-score"], report["1"]["f1-score"]]
    
    # Create bar chart for metrics
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(len(classes))
    
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1_score, width, label="F1-Score")
    
    ax.set_xlabel("Classes")
    ax.set_ylabel("Scores")
    ax.set_title("Model Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    st.pyplot(fig)

# Plot Radar Chart for Player
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

# Plot Comparison Bar Chart
def plot_comparison_bar_chart(df, players, metrics):
    filtered_df = df[df['Name'].isin(players)]
    melted_df = filtered_df.melt(id_vars='Name', value_vars=metrics)
    sns.barplot(data=melted_df, x='variable', y='value', hue='Name')
    plt.title("Player Comparison")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Initialize Session State
if "model" not in st.session_state:
    st.session_state.model = None

if "report" not in st.session_state:
    st.session_state.report = None

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
    selected_player = st.selectbox("Select a Player for Radar Chart", options=filtered_data['Name'].unique())
    if selected_player:
        plot_radar_chart(filtered_data, selected_player)

    # User Choice for Training Dataset
    st.sidebar.subheader("Model Training")
    train_on_filtered = st.sidebar.checkbox("Train Model on Filtered Data", value=False)

    # Machine Learning: Train and Save Model
    st.subheader("Player Potential Prediction")
    if st.button("Train ML Model"):
        if train_on_filtered:
            if filtered_data.empty:
                st.warning("No data available after applying filters. Please adjust the filters.")
            else:
                st.write("Training the model on the filtered data...")
                model, report = train_ml_model(filtered_data)
        else:
            st.write("Training the model on the entire dataset...")
            model, report = train_ml_model(data)
        
        st.session_state.model = model  # Save model to session state
        st.session_state.report = report  # Save report to session state
        st.write("Model Training Completed!")
        plot_metrics(report)

    # Definitions Section
    st.subheader("What Do These Metrics Mean?")
    st.markdown("""
    - **Precision**: Out of the players predicted to have high potential, how many actually do.
    - **Recall**: Out of all actual high-potential players, how many did the model correctly identify.
    - **F1-Score**: A balance between precision and recall.
    - **Support**: The number of actual instances for each class in the test data.
    """)

    # Display Previous Training Results
    if st.session_state.report is not None:
        st.subheader("Previous Training Results")
        plot_metrics(st.session_state.report)

    # Allow predictions for individual players
    if st.session_state.model is not None:
        st.subheader("Predict Player Potential")
        selected_player = st.selectbox("Select a Player to Predict Potential", options=data['Name'].unique())
        if selected_player:
            player_data = data[data['Name'] == selected_player]
            potential = predict_potential(st.session_state.model, player_data)
            st.write(f"Predicted Potential for {selected_player}: {'High' if potential[0] == 1 else 'Low'}")
    else:
        st.warning("Please train the model first before making predictions.")

    # Export Reports
    st.subheader("Export Report")
    if st.button("Download CSV"):
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="scouting_report.csv", mime='text/csv')

    if st.button("Download PDF"):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Scouting Report", ln=True, align='C')
        for index, row in filtered_data.iterrows():
            pdf.ln(10)
            for col, value in row.items():
                pdf.cell(200, 10, txt=f"{col}: {value}", ln=True)
        pdf_data = pdf.output(dest="S").encode("latin1")
        st.download_button("Download PDF", data=pdf_data, file_name="scouting_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
