input_s = [
  "5 5",
  ".....",
  "..#..",
  ".....",
  "#.##.",
  ".#...",
  "5",
  "1 3 3 3",
  "2 2 2 4",
  "1 1 5 5",
  "1 1 5 1",
  "4 2 1 4"
]

output_s = [
  5,
  5,
  8,
  "Impossible",
  5
]

def build_graph_tree(mat, sizes):
  graph = {}
  for row in range(sizes[0]):
    for col in range(sizes[1]):
      neighbour_list = []
      if row > 0:
        if mat[row-1][col] == '.':
          neighbour_list.append(f"{row-1} {col}") 
      if row < sizes[0]-1:
        if mat[row+1][col] == '.':
          neighbour_list.append(f"{row+1} {col}")
      if col > 0:
        if mat[row][col-1] == '.':
          neighbour_list.append(f"{row} {col-1}")
      if col < sizes[1]-1:
        if mat[row][col+1] == '.':
          neighbour_list.append(f"{row} {col+1}")

      graph[f"{row} {col}"] = neighbour_list
  return graph

def search(start, finish, graph):
    visited = [start]
    paths = [[start]]

    if start == finish:
        return ""

    while paths:
        path = paths.pop(0)
        node = path[-1]

        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.append(neighbour)
                new_path = list(path)
                new_path.append(neighbour)
                paths.append(new_path)

            if neighbour == finish:
                return new_path
    
    # If while loop runs out and no returns were issued:
    return "Impossible"

def convert(path):
    if path == "Impossible":
        return path

    coords = [int(c) for coord in path for c in coord.split()]
    print(coords)
    out = ""
    for i in range(len(path)-1):
        d_row = coords[i*2+2] - coords[i*2]
        d_col = coords[i*2+3] - coords[i*2+1]
        if d_row > 0:
            out += 'D'
        if d_row < 0:
            out += 'U'
        if d_col > 0:
            out += 'R'
        if d_col < 0:
            out += 'L'
    return out

def find_path(input_s):
  sizes = [int(s) for s in input_s[0].split()]
  mat = input_s[1:1+sizes[0]]
  graph = build_graph_tree(mat, sizes)

  out = []
  for line in input_s[2+sizes[0]:]:
    sx, sy, fx, fy = [int(s)-1 for s in line.split()]
    out.append(convert(search(f"{sx} {sy}", f"{fx} {fy}", graph)))
  return out

print(f"answer: {find_path(input_s)}")