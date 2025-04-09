import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from textblob import TextBlob
import umap

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load data
def load_data(file_path):
    """
    Load data from Excel file and preprocess the feedback column
    """
    df = pd.read_excel(file_path)
    
    # Assuming column BA is at index 52 (0-based)
    feedback_col_idx = 52
    col_name = df.columns[feedback_col_idx]
    
    print(f"Processing feedback from column: {col_name}")
    
    # Clean and filter data
    df['clean_feedback'] = df[col_name].astype(str).apply(lambda x: clean_text(x))
    valid_df = df[df['clean_feedback'].apply(lambda x: len(x) > 5)].copy()
    
    print(f"Total rows: {len(df)}, Valid feedback: {len(valid_df)}")
    
    return valid_df

def clean_text(text):
    """
    Clean and normalize text
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Remove null-like responses
    if text.lower() in ['nil', 'n/a', 'na', '-nil-', 'none', '-', 'null', 'n.a.', 'n a']:
        return ""
    
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s.,;!?\'"-]', '', text)  # Keep only alphanumeric and basic punctuation
    
    return text.strip()

# Advanced text preprocessing
def preprocess_text(texts, min_word_length=2):
    """
    Advanced text preprocessing including tokenization, stop word removal, and lemmatization
    """
    stop_words = set(stopwords.words('english'))
    custom_stops = {'would', 'could', 'should', 'may', 'also', 'us', 'one', 'will', 'please', 'think', 'get',
                   'make', 'etc', 'lot', 'way', 'even', 'much', 'many', 'really', 'use'}
    stop_words.update(custom_stops)
    
    lemmatizer = WordNetLemmatizer()
    
    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing texts"):
        if not text or len(text) < 5:
            processed_texts.append([])
            continue
            
        # Tokenize and lowercase
        tokens = simple_preprocess(text, deacc=True, min_len=min_word_length)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        processed_texts.append(tokens)
    
    return processed_texts

# Topic Modeling
def perform_topic_modeling(processed_docs, num_topics=10, method='lda'):
    """
    Perform topic modeling using either LDA or NMF
    """
    # Create dictionary and corpus
    dictionary = Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Choose method
    if method.lower() == 'lda':
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='auto',
            random_state=42
        )
    else:  # NMF
        tfidf = TfidfVectorizer(max_df=0.7, min_df=5, stop_words='english')
        doc_texts = [' '.join(doc) for doc in processed_docs]
        tfidf_matrix = tfidf.fit_transform(doc_texts)
        model = NMF(n_components=num_topics, random_state=42)
        model.fit(tfidf_matrix)
        
        # Create a wrapper for NMF to match LDA interface
        class NMFWrapper:
            def __init__(self, nmf_model, vectorizer):
                self.model = nmf_model
                self.vectorizer = vectorizer
                self.feature_names = vectorizer.get_feature_names_out()
            
            def print_topics(self, num_words=10):
                topics = []
                for topic_idx, topic in enumerate(self.model.components_):
                    top_features_ind = topic.argsort()[:-num_words-1:-1]
                    top_features = [self.feature_names[i] for i in top_features_ind]
                    weights = [topic[i] for i in top_features_ind]
                    topics.append((topic_idx, [(word, weight) for word, weight in zip(top_features, weights)]))
                return topics
                
        model = NMFWrapper(model, tfidf)
    
    # Evaluate model coherence
    if method.lower() == 'lda':
        coherence_model = CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"Topic Coherence Score: {coherence_score}")
    
    # Get topics
    topics = model.print_topics(num_words=15)
    for topic_id, topic_terms in topics:
        if method.lower() == 'lda':
            terms = " + ".join([f"{term}*{round(weight, 3)}" for term, weight in topic_terms])
        else:
            terms = " + ".join([f"{term}*{round(weight, 3)}" for term, weight in topic_terms])
        print(f"Topic {topic_id}: {terms}")
    
    # Assign topics to documents
    if method.lower() == 'lda':
        doc_topics = [max(model[corpus[i]], key=lambda x: x[1])[0] if corpus[i] else -1 for i in range(len(corpus))]
    else:
        doc_texts = [' '.join(doc) for doc in processed_docs]
        doc_vectors = model.vectorizer.transform(doc_texts)
        doc_topics = model.model.transform(doc_vectors).argmax(axis=1)
    
    return model, doc_topics, dictionary, corpus

# Embedding-based analysis
def create_document_embeddings(texts):
    """
    Create document embeddings using SentenceTransformer
    """
    print("Generating document embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model that works well
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def cluster_embeddings(embeddings, n_clusters=8):
    """
    Cluster document embeddings
    """
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    K_range = range(2, min(20, len(embeddings) // 10))
    
    print("Finding optimal number of clusters...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette score: {score:.4f}")
    
    # Plotting silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(K_range), silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette scores for different numbers of clusters')
    plt.grid(True)
    plt.savefig('silhouette_scores.png')
    
    # Choose optimal number of clusters
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform final clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, optimal_k

def visualize_embeddings(embeddings, labels, texts, method='tsne'):
    """
    Visualize document embeddings in 2D
    """
    print(f"Creating {method.upper()} visualization...")
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 10))
    else:  # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=min(30, len(embeddings) // 5))
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab20', alpha=0.7, s=40)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Document Embedding Visualization using {method.upper()}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(f'embedding_visualization_{method}.png')
    
    return embedding_2d

def analyze_clusters(texts, labels, processed_docs, n_clusters):
    """
    Analyze the content of each cluster
    """
    cluster_analysis = {}
    
    for cluster_id in range(n_clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
        cluster_tokens = [processed_docs[i] for i in range(len(processed_docs)) if labels[i] == cluster_id]
        
        if not cluster_texts:
            continue
            
        # Get top words in cluster
        all_words = [word for doc in cluster_tokens for word in doc]
        word_freq = nltk.FreqDist(all_words)
        top_words = word_freq.most_common(20)
        
        # Get representative examples
        sample_size = min(5, len(cluster_texts))
        examples = np.random.choice(cluster_texts, sample_size, replace=False).tolist()
        
        # Generate word cloud
        if all_words:
            wordcloud = WordCloud(
                background_color='white',
                max_words=50,
                max_font_size=100,
                width=800,
                height=400,
                random_state=42
            ).generate(' '.join(all_words))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Cluster {cluster_id} Word Cloud')
            plt.tight_layout()
            plt.savefig(f'cluster_{cluster_id}_wordcloud.png')
        
        # Store cluster info
        cluster_analysis[cluster_id] = {
            'size': len(cluster_texts),
            'top_words': top_words,
            'examples': examples
        }
        
        print(f"\nCluster {cluster_id} ({len(cluster_texts)} documents):")
        print(f"Top words: {', '.join([word for word, count in top_words[:10]])}")
        print("Example feedback:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example[:100]}..." if len(example) > 100 else f"  {i}. {example}")
    
    return cluster_analysis

# Sentiment Analysis
def perform_sentiment_analysis(texts):
    """
    Perform sentiment analysis on feedback texts
    """
    sentiments = []
    
    for text in tqdm(texts, desc="Analyzing sentiment"):
        blob = TextBlob(text)
        sentiments.append({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'text': text
        })
    
    sentiments_df = pd.DataFrame(sentiments)
    
    # Categorize sentiment
    sentiments_df['sentiment_category'] = pd.cut(
        sentiments_df['polarity'],
        bins=[-1.01, -0.2, 0.2, 1.01],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # Visualize sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=sentiments_df, x='sentiment_category')
    plt.title('Sentiment Distribution in Feedback')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    
    # Display sentiment examples
    print("\nSentiment Analysis Examples:")
    for category in ['Positive', 'Neutral', 'Negative']:
        category_df = sentiments_df[sentiments_df['sentiment_category'] == category]
        if not category_df.empty:
            examples = category_df.sample(min(3, len(category_df)))
            print(f"\n{category} examples:")
            for _, example in examples.iterrows():
                print(f"  - '{example['text'][:100]}...' (Polarity: {example['polarity']:.2f})" if len(example['text']) > 100 
                     else f"  - '{example['text']}' (Polarity: {example['polarity']:.2f})")
    
    return sentiments_df

# Named Entity Recognition
def perform_ner_analysis(texts):
    """
    Perform Named Entity Recognition to identify key entities
    """
    entities = []
    
    for text in tqdm(texts[:500], desc="Performing NER"):  # Limiting to 500 texts for performance
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'context': text
            })
    
    entities_df = pd.DataFrame(entities)
    
    if not entities_df.empty:
        # Count entities by type
        entity_counts = entities_df['label'].value_counts()
        
        # Visualize entity types
        plt.figure(figsize=(12, 6))
        sns.barplot(x=entity_counts.index, y=entity_counts.values)
        plt.title('Named Entity Types in Feedback')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('entity_types.png')
        
        # Display common entities
        print("\nCommon Named Entities:")
        for label in entity_counts.index[:5]:  # Top 5 entity types
            label_entities = entities_df[entities_df['label'] == label]
            common_entities = label_entities['text'].value_counts().head(5)
            print(f"\n{label} entities:")
            for entity, count in common_entities.items():
                print(f"  - '{entity}': {count} occurrences")
    else:
        print("No entities found in the analysis")
    
    return entities_df

# Main function
def analyze_feedback(file_path):
    """
    Comprehensive NLP analysis of feedback data
    """
    print("=== Starting Feedback Analysis ===")
    
    # Load and preprocess data
    df = load_data(file_path)
    valid_feedback = df['clean_feedback'].tolist()
    
    # Text preprocessing
    processed_docs = preprocess_text(valid_feedback)
    
    # Document Embeddings and Clustering
    embeddings = create_document_embeddings(valid_feedback)
    cluster_labels, n_clusters = cluster_embeddings(embeddings)
    embedding_2d = visualize_embeddings(embeddings, cluster_labels, valid_feedback, method='umap')
    cluster_analysis = analyze_clusters(valid_feedback, cluster_labels, processed_docs, n_clusters)
    
    # Topic Modeling
    print("\n=== Topic Modeling Analysis ===")
    model, doc_topics, dictionary, corpus = perform_topic_modeling(processed_docs, num_topics=10, method='lda')
    
    # Add topic and cluster info to the dataframe
    df['topic_id'] = doc_topics
    df['cluster_id'] = cluster_labels
    
    # Sentiment Analysis
    print("\n=== Sentiment Analysis ===")
    sentiment_df = perform_sentiment_analysis(valid_feedback)
    df['sentiment_polarity'] = sentiment_df['polarity']
    df['sentiment_category'] = sentiment_df['sentiment_category']
    
    # Named Entity Recognition
    print("\n=== Named Entity Recognition ===")
    entities_df = perform_ner_analysis(valid_feedback)
    
    # Save results
    df.to_excel('feedback_analysis_results.xlsx', index=False)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to 'feedback_analysis_results.xlsx'")
    
    return df, cluster_analysis, embeddings, embedding_2d, cluster_labels

# Combined plots
def create_summary_dashboard(df, cluster_analysis, embedding_2d, cluster_labels):
    """
    Create summary visualizations combining multiple analysis types
    """
    # Create a dashboard with multiple plots
    plt.figure(figsize=(20, 15))
    
    # 1. Embedding visualization with sentiment coloring
    plt.subplot(2, 2, 1)
    sentiment_colors = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'}
    sentiment_numeric = df['sentiment_category'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=sentiment_numeric, 
                cmap=plt.cm.get_cmap('RdYlGn', 3), alpha=0.7, s=30)
    plt.colorbar(ticks=[0, 1, 2], label='Sentiment')
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.title('Document Map Colored by Sentiment')
    
    # 2. Cluster distribution
    plt.subplot(2, 2, 2)
    cluster_counts = df['cluster_id'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    plt.title('Feedback Distribution by Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Feedback Items')
    
    # 3. Sentiment distribution by cluster
    plt.subplot(2, 2, 3)
    cluster_sentiment = df.groupby('cluster_id')['sentiment_polarity'].mean().sort_index()
    sns.barplot(x=cluster_sentiment.index, y=cluster_sentiment.values)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Average Sentiment by Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Sentiment Polarity')
    
    # 4. Topic distribution
    plt.subplot(2, 2, 4)
    topic_counts = df['topic_id'].value_counts().sort_index()
    sns.barplot(x=topic_counts.index, y=topic_counts.values)
    plt.title('Feedback Distribution by Topic')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Feedback Items')
    
    plt.tight_layout()
    plt.savefig('feedback_analysis_dashboard.png', dpi=300)
    
    # Create a summary report
    with open('feedback_analysis_summary.md', 'w') as f:
        f.write("# Feedback Analysis Summary\n\n")
        
        f.write("## Cluster Analysis\n\n")
        for cluster_id, data in cluster_analysis.items():
            f.write(f"### Cluster {cluster_id} ({data['size']} items)\n\n")
            f.write("**Top Words:**\n")
            f.write(", ".join([f"{word} ({count})" for word, count in data['top_words'][:10]]) + "\n\n")
            f.write("**Example Feedback:**\n")
            for i, example in enumerate(data['examples'][:3], 1):
                f.write(f"{i}. {example[:200]}...\n" if len(example) > 200 else f"{i}. {example}\n")
            f.write("\n")
        
        f.write("## Sentiment Analysis\n\n")
        sentiment_counts = df['sentiment_category'].value_counts()
        f.write("**Sentiment Distribution:**\n")
        for category, count in sentiment_counts.items():
            f.write(f"- {category}: {count} items ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("## Topic Analysis\n\n")
        for topic_id in sorted(df['topic_id'].unique()):
            topic_size = len(df[df['topic_id'] == topic_id])
            f.write(f"### Topic {topic_id} ({topic_size} items)\n\n")
            topic_examples = df[df['topic_id'] == topic_id]['clean_feedback'].sample(min(3, topic_size)).tolist()
            f.write("**Example Feedback:**\n")
            for i, example in enumerate(topic_examples, 1):
                f.write(f"{i}. {example[:200]}...\n" if len(example) > 200 else f"{i}. {example}\n")
            f.write("\n")
    
    print("Summary dashboard and report created!")


if __name__ == "__main__":

    file_path = "C:\\alexyeapcl\\Admin\\Documents\\VS Code Projects\\CCL Wayfinding\\Public Engagement on Enhanced Circle Line Wayfinding Designs.xlsx"

    df, cluster_analysis, embeddings, embedding_2d, cluster_labels = analyze_feedback(file_path)

    create_summary_dashboard(df, cluster_analysis, embedding_2d, cluster_labels)
