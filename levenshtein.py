def levenshtein_distance(s, t, sub_cost=1):
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    for row in range(1, rows): dist[row][0] = row
    for col in range(1, cols): dist[0][col] = col    
    for col in range(1, cols):
        for row in range(1, rows):
            dist[row][col] = min(dist[row-1][col] + 1, dist[row][col-1] + 1, dist[row-1][col-1] + (0 if s[row-1] == t[col-1] else sub_cost))
    return dist[row][col]