import pandas as pd

file_path = "soc-redditHyperlinks-title.tsv"
df = pd.read_csv(file_path, sep='\t')
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
df = df[df['TIMESTAMP'].notna()]

#Clean it
df.drop_duplicates(inplace=True)

property_names = [
    "Num_Chars", "Num_Chars_NoSpace", "Frac_Alpha", "Frac_Digits", "Frac_Upper",
    "Frac_Whitespace", "Frac_Special", "Num_Words", "Num_Unique_Words",
    "Num_Long_Words", "Avg_Word_Length", "Num_Unique_Stopwords", "Frac_Stopwords",
    "Num_Sentences", "Num_Long_Sentences", "Avg_Chars_Per_Sentence",
    "Avg_Words_Per_Sentence", "Readability_Index", "Sent_Positive", "Sent_Negative",
    "Sent_Compound", "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I",
    "LIWC_We", "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron",
    "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present",
    "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate",
    "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
    "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo",
    "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight",
    "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib",
    "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear",
    "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual",
    "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time",
    "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money",
    "LIWC_Relig", "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu",
    "LIWC_Filler"
]

properties_expanded = df['PROPERTIES'].str.split(',', expand=True)
properties_expanded = properties_expanded.iloc[:, :len(property_names)]  # Ensure consistent length
properties_expanded.columns = property_names
properties_expanded = properties_expanded.apply(pd.to_numeric, errors='coerce')
df_clean = pd.concat([df.drop(columns=['PROPERTIES']), properties_expanded], axis=1)
df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)



df_clean.to_csv("soc-redditHyperlinks-title_clean.csv", index=False)
print("\nCleaned dataset saved as 'soc-redditHyperlinks-title_clean.csv'")