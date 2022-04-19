from sentence_transformers import SentenceTransformer
import pickle
import re



def generate(df):
    sentences = df['lemmas'].tolist()
    embeddings = model.encode(sentences)
    df['curr_enc'] = ''
    for i, el in enumerate(embeddings):
        df['curr_enc'][i] = el
    
    return df


if __name__ == "__main__":
    import time
    tmp = time.time()
    with open('/root/sanchit/research-group/data/val.pkl', 'rb') as f:
        df = pickle.load(f)
    model = SentenceTransformer('all-mpnet-base-v2')
    print("Model and data loaded")
    df = generate(df)
    print(df.iloc[0])
    with open('data/val_enc.pkl', 'wb') as f:
        pickle.dump(df, f)
    print(f"Time taken: {time.time() - tmp}")