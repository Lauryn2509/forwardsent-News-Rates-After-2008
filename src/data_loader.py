import pandas as pd


def load_fed_rates_from_csv(filepath: str) -> pd.DataFrame:
    """
    Charge les taux de la Fed à partir d'un fichier CSV local de type FRED.
    Le fichier doit contenir les colonnes : 'observation_date' et 'FEDFUNDS'.
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    df.rename(columns={"observation_date": "Date", "FEDFUNDS": "rate"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "rate"]).reset_index(drop=True)
    return df


def load_headlines_from_csv(filepath: str) -> pd.DataFrame:
    """
    Charge des titres de presse depuis un fichier CSV type 'stocknews.csv' ou Reddit.
    Fusionne les colonnes Top1 à Top25 en une seule colonne 'headlines' par jour.
    """
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    # Décodage éventuel des bytes
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    # Harmonisation du nom de la colonne date
    if "date" in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)

    # Nettoyage + conversion automatique robuste
    df["Date"] = df["Date"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove timezone info if present to avoid issues
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    # Sélection des colonnes Top1 à Top25 et concaténation, ou utilisation directe si 'headlines' existe
    if "headlines" in df.columns:
        # Si la colonne headlines existe déjà, on l'utilise directement
        df["headlines"] = df["headlines"].astype(str).fillna("")
    else:
        # Sinon, on concatène les colonnes Top1 à Top25
        top_cols = [col for col in df.columns if col.lower().startswith("top")]
        if not top_cols:
            raise ValueError(
                "Aucune colonne 'headlines' ou 'TopX' trouvée dans le fichier"
            )
        df["headlines"] = df[top_cols].apply(
            lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1
        )

    # Nettoyage final
    df = df[["Date", "headlines"]].dropna(subset=["Date", "headlines"])
    df = df[df["headlines"].str.strip() != ""]  # Remove empty headlines

    return df


def build_dataset(fed_csv_path, headlines_csv_path=None, rss_urls=None, save_path=None):
    """
    Étapes 1 à 3 : Charge les taux de la Fed + headlines (CSV ou RSS),
    nettoie les titres et retourne un DataFrame joint, groupé par mois.
    """
    from src.text_cleaner import fetch_rss_titles

    # Étape 1 : Chargement des taux FED
    df_rates = load_fed_rates_from_csv(fed_csv_path)
    df_rates["Date"] = (
        pd.to_datetime(df_rates["Date"]).dt.to_period("M").dt.to_timestamp()
    )

    # Étape 2 : Chargement des headlines
    if headlines_csv_path:
        df_news = load_headlines_from_csv(headlines_csv_path)
    elif rss_urls:
        df_news = fetch_rss_titles(rss_urls)
        df_news.rename(columns={"title": "headlines"}, inplace=True)
    else:
        raise ValueError("Fournir soit headlines_csv_path, soit rss_urls")

    # Convertir la date en début de mois (période mensuelle)
    df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")

    # Remove rows with NaT dates
    df_news = df_news.dropna(subset=["Date"])

    df_news["Month"] = df_news["Date"].dt.to_period("M")  # YYYY-MM

    # Grouper tous les titres du mois en une seule chaîne
    df_grouped = (
        df_news.groupby("Month")["headlines"]
        .agg(lambda x: " ".join(x.dropna()))
        .reset_index()
    )

    # Convertir la période "Month" en timestamp (YYYY-MM-01)
    df_grouped["Date"] = df_grouped["Month"].dt.to_timestamp()
    print(df_grouped.head())  # Afficher les premières lignes pour vérification
    df_grouped = df_grouped[["Date", "headlines"]]  # réorganiser

    # Fusionner avec les taux FED
    df_merged = pd.merge(df_grouped, df_rates, on="Date", how="inner")

    # Rename Date to date for consistency with main.py
    df_merged.rename(columns={"Date": "date", "rate": "fed_rate"}, inplace=True)

    # Sauvegarde éventuelle
    if save_path:
        df_merged.to_csv(save_path, index=False)
        print(f"✅ Dataset fusionné sauvegardé : {save_path}")
        print(
            f"🔎 {len(df_merged)} lignes fusionnées sur {len(df_grouped)} mois de headlines et {len(df_rates)} taux."
        )

    return df_merged
