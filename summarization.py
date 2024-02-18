import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from embedding import get_embedding

embeddings = get_embedding()



def text_rank(text, embeddings):
    sum_size = int(input('How many sentences to extract?'))
    sim_mat = cosine_similarity(embeddings)# вычисление матрицы косинусового растояния между предложениями
    scaler = MinMaxScaler(feature_range=(0, 1))# нормализация значений в пределах от 0 до 1
    sim_mat = scaler.fit_transform(
        sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings),
                                                  len(embeddings)
                                                  )# нормализация значений в пределах от 0 до 1
    nx_graph = nx.from_numpy_array(sim_mat)# создание графа на основе матрицы косинусового подобия
    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=500)# алгоритм pagerank
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(text)), reverse=True)
    sum_result = []
    for i in range(sum_size):
        sum_result.append(ranked_sentences[i][1])
    return sum_result



def find_nearest_sentences(centroid, candidates, n=1):
    """
    :parameter centroid: n-dim vector
    :parameter candidates: list of triplets (sentence, embedding)
    :parameter n: how many sentences to extract
    """
    nearest = []
    for can, emb in candidates:
        if type(emb) == np.ndarray:
            score = cosine_similarity(centroid.reshape(1, -1), emb.reshape(1, -1))
        else:  # convert to numpy
            score = cosine_similarity(centroid.reshape(1, -1), emb.numpy().reshape(1, -1))
        nearest.append((score, can))

    nearest.sort(reverse=True)
    return [nearest[i][1] for i in range(n)]


def kmeans_summary(text, embeddings):
    sum_size = int(input('How many sentences to extract?'))
    n_clusters = int(input('Enter number of clusters: '))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embeddings)
    labels = list(kmeans.labels_)
    extracted = []
    for idx, centroid in enumerate(kmeans.cluster_centers_):
        sent_embedding = [
            (sent, embedding) for sent, embedding, cluster in zip(text, embeddings, labels) if cluster == idx
        ]
        for sent in find_nearest_sentences(centroid, sent_embedding, n=sum_size):
            extracted.append(sent)
    return extracted


def scale_centrality_scores(centrality_scores, q=0.1):
    scaler = MinMaxScaler(feature_range=(1, 10))
    scaler.fit(centrality_scores[centrality_scores >= np.quantile(centrality_scores, q=q)].reshape(-1, 1))
    centrality_scores = scaler.transform(centrality_scores.reshape(-1, 1))
    centrality_scores = np.where(centrality_scores < 0, 0.5, centrality_scores).ravel()
    return centrality_scores


def text_rank_visual(comments):
    perf = comments.ChannelName.unique().tolist()
    for index, item in enumerate(perf):
        print(f'{index + 1}. {item}')
    if len(perf) > 1:
        numb = int(input('Enter channel title number:'))
        channel_name = perf[numb - 1]
    else:
        channel_name = perf[0]
    emotional_colors = ['positive', 'negative', 'neutral']
    numb = int(input('Choose sentiment:\n1. positive\n2. negative\n3. neutral\n4. all'))
    sentiment = emotional_colors[numb - 1]
    sentences = comments.loc[
        (comments.Language == 'en') & (comments.ChannelName == channel_name) & (comments.Emotional == sentiment),
        'Comment'
    ].tolist()


    choose = int(input('Choose encoder type:\n1. SBERT\n2. LABSE'))
    if choose == 1:
        embeddings = sbert_model.encode(sentences=sentences, convert_to_tensor=True)
    elif choose == 2:
        embeddings = labse_model.encode(sentences=sentences)

    # similarity matrix
    sim_mat = cosine_similarity(embeddings)

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))

    # calculate pagerank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=500)  # number of cycles to converge
    score_list = [scores[sent_idx] for sent_idx in range(len(sentences))]

    # reduce dimensionality
    pca = PCA(n_components=2)
    pos = pca.fit_transform(embeddings)

    # get weights
    weights = sim_mat
    centrality_scores = np.array(score_list)
    centrality_scores = scale_centrality_scores(centrality_scores, q=0.1)
    np.fill_diagonal(weights, 0)

    G = nx.from_numpy_array(weights)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if weights[edge[0], edge[1]] > 0.5:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.75,
                  color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            # reversescale=True,
            color=[],
            size=[s * 10 for s in centrality_scores],
            colorbar=dict(
                thickness=15,
                title='Centrality Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    node_adjacencies = []
    node_text = []
    for node, weight in enumerate(centrality_scores):
        node_adjacencies.append(weight)
        node_text.append(sentences[node])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<b>TextRank Summarization</b>',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                    ))
    fig.show()


def kmeans_clustering(comments):
    perf = comments.ChannelName.unique().tolist()
    for index, item in enumerate(perf):
        print(f'{index + 1}. {item}')
    if len(perf) > 1:
        numb = int(input('Enter channel title number:'))
        channel_name = perf[numb-1]
    else:
        channel_name = perf[0]
    emotional_colors = ['positive', 'negative', 'neutral']
    numb = int(input('Choose sentiment:\n1. positive\n2. negative\n3. neutral\n4. all'))
    sentiment = emotional_colors[numb - 1]
    sentences = comments.loc[
        (comments.Language == 'en') & (comments.ChannelName == channel_name) & (comments.Emotional == sentiment),
        'Comment'
    ].tolist()

    choose = int(input('Choose encoder type:\n1. SBERT\n2. LABSE'))
    if choose == 1:
        embeddings = sbert_model.encode(sentences=sentences, convert_to_tensor=True)
    elif choose == 2:
        embeddings = labse_model.encode(sentences=sentences)
    pca = PCA(n_components=2)
    reduced_dim_embeddings = pca.fit_transform(embeddings)

    num_clusters = int(input('Enter number of clusters:'))
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(reduced_dim_embeddings)
    clustering = kmeans.labels_


    xs = [x for x, _ in reduced_dim_embeddings]
    ys = [y for _, y in reduced_dim_embeddings]
    labels = [f'Cluster {c}' for c in clustering]

    df = pd.DataFrame(
        {
            'x': xs,
            'y': ys,
            'cluster': labels,
            'sentences': sentences
        }
    )

    # визуализация с помощью Plotly
    fig = px.scatter(df,
                     x='x',
                     y='y',
                     hover_name='sentences',
                     color='cluster',
                     title='<b>Dimensionality reduced by PCA, colored with KMeans</b>'
                     )

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.add_trace(
        go.Scatter(
            x=[x for x, _ in kmeans.cluster_centers_],
            y=[y for _, y in kmeans.cluster_centers_],
            showlegend=False,
            hovertext=list(range(len(kmeans.cluster_centers_))),
            mode='markers',
            marker=dict(
                color='Yellow',
                size=16,
                symbol='x',
            )
        )
    )
    fig.show()
