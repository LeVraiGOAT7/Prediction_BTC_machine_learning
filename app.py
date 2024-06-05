import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import holidays

def main():
    st.title("Application de prédiction de prix des cryptomonnaies (Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB))")
    
    st.subheader("Auteurs: [BIBANG Emmanuel](https://www.linkedin.com/in/emmanuel-marvin-marvin-childerick-brinest-bibang-77395b216?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) & [Jean JUSTINE](https://www.linkedin.com/in/jean-justine/)")

    # Ajouter une image sous le titre
    st.image("image_crypto.webp", use_column_width=True)

    # Ajouter une légende avec une police augmentée
    legend_html = """
    <div style="text-align: center;">
        <span style="font-size: 24px;">Pour de meilleurs choix d'investissement</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    st.image("image_prediction.gif", use_column_width=True)

    # Fonction d'importation des données
    @st.cache_data(persist=True) 
    def load_data(coin):
        data = pd.read_csv(f'{coin}_price.csv', index_col=0)  # Charger les données avec l'index en place
        data.reset_index(inplace=True)  # Réinitialiser l'index pour le transformer en colonne
        data.rename(columns={'index': 'Date'}, inplace=True)  # Renommer la nouvelle colonne en 'Date'
        data['Date'] = pd.to_datetime(data['Date'])  # Convertir la colonne 'Date' au format datetime
        return data 
    
    # Sélection de la cryptomonnaie
    coin_selection = st.sidebar.selectbox("Sélectionner la cryptomonnaie", ["Bitcoin (BTC)", "Ethereum (ETH)", "Binance Coin (BNB)"])
    coin = coin_selection.split()[0].lower()  # Récupérer le nom de la cryptomonnaie en minuscules sans l'abréviation
    
    # Affichage de la table de données
    df = load_data(coin)
    df_sample = df.sample(100)
    if st.sidebar.checkbox(f"Afficher les données brutes de {coin_selection}", False):
        st.subheader(f"Jeu de données prix de clôture de {coin_selection}: Echantillon de 100 observations")
        st.write(df_sample)
        st.line_chart(df['Close'], use_container_width=True)
        fig, ax = plt.subplots()
        ax.hist(df.Close)
        st.pyplot(fig)
    seed = 123

    st.sidebar.subheader("Modèle de prédiction")
    
    # Saisir les dates de début et de fin avec restrictions
    start_date = st.sidebar.date_input(
        "Saisir la date d'entrée (Start date)", 
        value=pd.to_datetime('2023-11-30'), 
        min_value=pd.to_datetime('2023-11-30')
    )
    
    max_end_date = start_date + pd.DateOffset(years=2)
    
    end_date = st.sidebar.date_input(
        "Saisir la date de sortie (End date)", 
        value=start_date + pd.DateOffset(months=1),
        min_value=start_date,
        max_value=max_end_date
    )

    # Bouton pour lancer la prédiction
    if st.sidebar.button("Lancer la prédiction"):
        # Préparation des données pour Prophet
        df['ds'] = df['Date']
        df['y'] = df['Close']
        
        # Convertir start_date en datetime64 pour assurer la compatibilité
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Sélection des données avant la date de début pour l'entraînement
        train_data = df[df['ds'] <= start_date]
        
        # Définition des jours fériés des États-Unis
        us_holidays = holidays.US(years=range(2010, 2025))  # Assurez-vous de couvrir l'intervalle de vos données
        
        holidays_df = pd.DataFrame(list(us_holidays.items()), columns=['ds', 'holiday'])
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

        # Création et entraînement du modèle Prophet avec les jours fériés et la saisonnalité
        model = Prophet(holidays=holidays_df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Saisonnalité mensuelle
        model.fit(train_data)
        
        # Création d'un DataFrame pour les futures prédictions
        future_dates = model.make_future_dataframe(periods=(end_date - start_date).days)
        
        # Prédictions
        forecast = model.predict(future_dates)
        
        # Filtrer les prédictions pour l'intervalle de dates spécifié
        prediction = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
        
        # Afficher les résultats de la prédiction
        st.write(f"Prédictions de prix de {coin_selection} entre", start_date, "et", end_date)
        st.write(prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        # Plot des prédictions
        fig, ax = plt.subplots()
        model.plot(forecast, ax=ax)
        ax.axvline(start_date, color='r', linestyle='--')
        ax.axvline(end_date, color='r', linestyle='--')
        st.pyplot(fig)

if __name__ == "__main__":
    main()

