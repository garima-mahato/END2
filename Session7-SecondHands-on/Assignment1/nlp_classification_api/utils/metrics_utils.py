from sklearn.metrics import f1_score, accuracy_score

def print_accuracy(df, target_col, pred_column):
    "Print f1 score and accuracy after making predictions"
    f1_macro = f1_score(df[target_col].astype(int), df[pred_column].astype(int), average='macro')
    acc = accuracy_score(df[target_col].astype(int), df[pred_column].astype(int))*100
    return f1_macro, acc